import argparse


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    existing = {a.option_strings[0] for a in parser._actions if a.option_strings}
    if "--cutmix_warmup_iter" not in existing:
        parser.add_argument(
            "--cutmix_warmup_iter", type=int, help="Number of warmup iterations for cutmix"
        )
    if "--cutmix_prob" not in existing:
        parser.add_argument("--cutmix_prob", type=float, help="Cutmix probability")
    if "--lam_adv" not in existing:
        parser.add_argument("--lam_adv", type=float, help="Adv. weight in generator loss")
    if "--lam_px_im" not in existing:
        parser.add_argument(
            "--lam_px_im", type=float, help="Pixelwise loss of image in generator loss"
        )
    if "--lam_px_grad" not in existing:
        parser.add_argument(
            "--lam_px_grad", type=float, help="Pixelwise loss of gradient in generator loss"
        )
    if "--lam_cutmix" not in existing:
        parser.add_argument(
            "--lam_cutmix", type=float, help="Weight of cutmix loss in distriminator loss"
        )
    return parser
