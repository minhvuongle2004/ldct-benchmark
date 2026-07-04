import argparse
import io
import os
import warnings
import zipfile

import pandas as pd
import requests
from tqdm import tqdm

# CRITICAL: Use v2 for authenticated endpoints
BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v2/"


def get_series(manifest_file):
    # Similar to tcia_utils.nbia.manifestToList()
    data = []
    with open(manifest_file) as f:
        for line in f:
            data.append(line.rstrip())
    del data[:6]  # remove header
    return data


def get_token(user, pwd):
    # Similar to tcia_utils.nbia.getToken()
    global api_call_headers
    try:
        params = {
            "username": user,
            "password": pwd,
            "client_id": "NBIA",
            "grant_type": "password",
        }

        # Updated: New OAuth token endpoint (without version)
        token_url = "https://services.cancerimagingarchive.net/nbia-api/oauth/token"
        
        data = requests.post(token_url, data=params, timeout=30)
        data.raise_for_status()
        access_token = data.json()["access_token"]
        api_call_headers = {"Authorization": "Bearer " + access_token}
        
        print(f"[OK] Authentication successful! Token obtained.")

    # handle errors
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {data.status_code}")
        print(f"Response: {data.text}")
        raise ValueError(
            f"HTTP Error: {data.status_code} -- Double check your user name and password."
        )
    except requests.exceptions.ConnectionError as errc:
        raise ValueError(f"Connection Error: Unable to connect to authentication server")
    except requests.exceptions.Timeout as errt:
        raise ValueError(f"Timeout Error: Authentication request timed out")
    except requests.exceptions.RequestException as err:
        raise ValueError(f"Request Error: {err}")


def download_series(series: str, savedir: str):
    # Similar to tcia_utils.nbia.downloadSeries()
    global metadata_df

    # Use v2 endpoints for authenticated access
    data_url = f"{BASE_URL}getImage?SeriesInstanceUID={series}"
    metadata_url = f"{BASE_URL}getSeriesMetaData?SeriesInstanceUID={series}"

    try:
        response = requests.get(metadata_url, headers=api_call_headers, timeout=30)
    except requests.exceptions.RequestException as e:
        print(f"Warning: Connection error for series {series}. Error: {e}. Skipping.")
        return

    # if the request was successful, get the metadata
    if response.status_code == 200:
        try:
            metadata = response.json()
            if not metadata or len(metadata) == 0:
                print(f"Warning: Empty metadata for series {series}. Skipping.")
                return
        except Exception as e:
            print(f"Warning: Failed to parse metadata for series {series}. Error: {e}. Skipping.")
            return
    elif response.status_code == 204:
        print(f"Warning: No content for series {series}. Skipping.")
        return
    elif response.status_code == 401:
        print(f"Warning: Unauthorized access for series {series}. Token may have expired. Skipping.")
        return
    elif response.status_code == 500:
        print(f"Warning: Server error for series {series}. The series may not exist or be inaccessible. Skipping.")
        return
    else:
        print(
            f"Warning: Received status code {response.status_code} for series {series}. Skipping."
        )
        return
    
    if (
        "Series UID" in metadata_df.columns
        and series in metadata_df["Series UID"].to_list()
    ):
        warnings.warn(f"Skip {series} as it was already downloaded")
        return

    # Construct folder path
    m = metadata[0]
    series_savedir = os.path.join(
        savedir,
        "LDCT-and-Projection-data",
        m["Subject ID"],
        f"{m['Study Date']}-NA-NA-{m['Study UID'][-5:]}",
        f"{m['Series Number']}-{m['Series Description']}-{m['Series UID'][-5:]}",
    )

    if not os.path.exists(series_savedir):
        os.makedirs(series_savedir)
    else:
        # Delete anything in the folder in case of a previous partial download
        for file in os.listdir(series_savedir):
            os.remove(os.path.join(series_savedir, file))

    # Download data
    try:
        data = requests.get(data_url, headers=api_call_headers, timeout=300)
        data.raise_for_status()
        
        if len(data.content) == 0:
            print(f"Warning: Empty content received for series {series}. Skipping.")
            return
            
        file = zipfile.ZipFile(io.BytesIO(data.content))
        file.extractall(path=series_savedir)
    except requests.exceptions.HTTPError as e:
        print(f"Warning: HTTP error downloading series {series}. Status: {data.status_code}. Skipping.")
        return
    except requests.exceptions.RequestException as e:
        print(f"Warning: Failed to download data for series {series}. Error: {e}. Skipping.")
        return
    except zipfile.BadZipFile as e:
        print(f"Warning: Invalid zip file for series {series}. Error: {e}. Skipping.")
        return

    # Update metadata
    metadata_df = pd.concat([metadata_df, pd.DataFrame(metadata)], ignore_index=True)


def main():
    # This script can be used to download the LDCT and Projection data using NBIA API.
    # Some of the code herein is heavily inspired by the tcia-utils python package
    # (https://github.com/kirbyju/tcia_utils). We do this to reduce package dependencies
    # (tcia_utils wants a lot of packages we don't need here), show progress bars, and
    # store data in same folder structure as the nbia-data-retriever would do.
    global metadata_df
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="", help="nbia manifest file")
    parser.add_argument(
        "--savedir", default="ldct-data", help="Folder to which data is downloaded"
    )
    parser.add_argument("--username", default="", help="TCIA username")
    parser.add_argument("--password", default="", help="TCIA password")
    opt = parser.parse_args()

    if not opt.manifest:
        opt.manifest = os.path.join("assets", "manifest.tcia")

    assert opt.username, "Must provide username since LDCT data has restricted access!"
    assert opt.password, "Must provide password since LDCT data has restricted access!"

    print("=" * 60)
    print("TCIA LDCT Data Downloader")
    print("=" * 60)
    
    # Get list of series we want to download
    series_to_download = get_series(opt.manifest)
    print(f"Found {len(series_to_download)} series in manifest file")

    # Get token since data access is restricted
    print("\nAuthenticating with TCIA...")
    get_token(user=opt.username, pwd=opt.password)

    os.makedirs(opt.savedir, exist_ok=True)

    # Create metadata csv
    metadata_path = os.path.join(opt.savedir, "metadata.csv")
    if os.path.isfile(metadata_path):
        print(f"Loading existing metadata from {metadata_path}")
        metadata_df = pd.read_csv(metadata_path)
        initial_rows = len(metadata_df)
        print(f"Found {initial_rows} previously downloaded series")
    else:
        metadata_df = pd.DataFrame()
        initial_rows = 0

    print("\nStarting download...\n")
    successful_downloads = 0
    failed_series = []
    
    for series in tqdm(series_to_download, desc="Download LDCT data"):
        # Download data
        rows_before = len(metadata_df)
        download_series(series=series, savedir=opt.savedir)
        if len(metadata_df) > rows_before:
            successful_downloads += 1
        else:
            failed_series.append(series)

    # After the loop, check if any *new* data was actually downloaded.
    if len(metadata_df) == initial_rows:
        print("\n" + "=" * 60)
        print("WARNING: No new data was downloaded.")
        print("=" * 60)
        print(
            f"All {len(series_to_download)} requested series failed to download."
        )
        print("\nPossible causes:")
        print("1. Incorrect username or password.")
        print("2. Account not granted access to 'LDCT-and-Projection-data'.")
        print("3. Series IDs in manifest file do not exist or have been deleted.")
        print("4. Token expired during download (tokens are valid for 2 hours).")
        print("\nSuggestions:")
        print("- Check access permissions at: https://www.cancerimagingarchive.net")
        print("- Contact TCIA support: help@cancerimagingarchive.net")
        print("=" * 60)
    else:
        # Update manifest file
        new_series_count = len(metadata_df) - initial_rows
        print("\n" + "=" * 60)
        print(f"[OK] Successfully downloaded {new_series_count} new series!")
        print("=" * 60)
        print(f"Success rate: {new_series_count}/{len(series_to_download)} ({100*new_series_count/len(series_to_download):.1f}%)")
        if failed_series:
            print(f"Failed to download {len(failed_series)} series (see warnings above)")
        print(f"\nSaving updated metadata to {metadata_path}")
        os.makedirs(opt.savedir, exist_ok=True)
        metadata_df.to_csv(metadata_path, index=False)
        print(f"Total series in metadata: {len(metadata_df)}")
        print("=" * 60)


if __name__ == "__main__":
    main()