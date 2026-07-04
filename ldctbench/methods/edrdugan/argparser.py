import argparse


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Thêm tham số riêng của EDR-DUGAN vào argument parser.

    Bao gồm:
      - Tham số DUGAN gốc: n_d_train, lam_adv, lam_px_im, lam_px_grad,
                           cutmix_prob, cutmix_warmup_iter, lam_cutmix
      - Tham số EDR: num_edge_blocks, use_sobel_input
    Dùng check-before-add để tránh conflict khi nhiều method trong METHODS list.
    """
    existing = {a.option_strings[0] for a in parser._actions if a.option_strings}

    # ── DUGAN-specific args ────────────────────────────────────────────────────
    if "--n_d_train" not in existing:
        parser.add_argument(
            "--n_d_train", type=int, default=2,
            help="Number of times to train D for each G iter (default = 2)"
        )
    if "--lam_adv" not in existing:
        parser.add_argument(
            "--lam_adv", type=float, default=0.08,
            help="Adversarial loss weight in Generator loss (default = 0.08)"
        )
    if "--lam_px_im" not in existing:
        parser.add_argument(
            "--lam_px_im", type=float, default=1.0,
            help="Pixelwise image loss weight in Generator loss (default = 1.0)"
        )
    if "--lam_px_grad" not in existing:
        parser.add_argument(
            "--lam_px_grad", type=float, default=27.8,
            help="Pixelwise gradient loss weight in Generator loss (default = 27.8)"
        )
    if "--lam_cutmix" not in existing:
        parser.add_argument(
            "--lam_cutmix", type=float, default=2.65,
            help="CutMix loss weight in Discriminator loss (default = 2.65)"
        )
    if "--cutmix_prob" not in existing:
        parser.add_argument(
            "--cutmix_prob", type=float, default=0.5,
            help="CutMix probability (default = 0.5)"
        )
    if "--cutmix_warmup_iter" not in existing:
        parser.add_argument(
            "--cutmix_warmup_iter", type=int, default=5000,
            help="Number of warmup iterations for CutMix (default = 5000)"
        )

    # ── EDR-specific args ──────────────────────────────────────────────────────
    if "--num_edge_blocks" not in existing:
        parser.add_argument(
            "--num_edge_blocks", type=int, default=2,
            help="Number of EdgeDilatedResidualBlocks at bottleneck (default = 2)"
        )
    if "--use_sobel_input" not in existing:
        parser.add_argument(
            "--use_sobel_input", action="store_true", default=True,
            help="Whether to use FixedSobelLayer features at bottleneck (default = True)"
        )
        parser.add_argument(
            "--no_sobel_input", action="store_false", dest="use_sobel_input",
            help="Disable FixedSobelLayer for ablation study"
        )

    return parser
