import argparse


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    existing = {a.option_strings[0] for a in parser._actions if a.option_strings}
    if "--n_d_train" not in existing:
        parser.add_argument(
            "--n_d_train", type=int, help="Number of times to train D for each G iter"
        )
    if "--lam_perc" not in existing:
        parser.add_argument(
            "--lam_perc", type=float, help="Weight of VGG (perceptual) loss"
        )
    return parser
