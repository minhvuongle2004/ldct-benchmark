import argparse


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """EDR-CNN10 dùng chung args với edrrednet.

    Để tránh conflict khi cả edrcnn10 và edrrednet cùng nằm trong METHODS,
    kiểm tra trước khi thêm từng argument.
    """
    existing = {a.option_strings[0] for a in parser._actions if a.option_strings}

    if "--loss_alpha" not in existing:
        parser.add_argument(
            "--loss_alpha", type=float, default=0.1,
            help="Weight for SobelEdgeLoss in CombinedLoss (default=0.1)",
        )
    if "--loss_beta" not in existing:
        parser.add_argument(
            "--loss_beta", type=float, default=0.0,
            help="Weight for Perceptual (VGG) Loss (default=0.0=OFF)",
        )
    if "--loss_gamma" not in existing:
        parser.add_argument(
            "--loss_gamma", type=float, default=0.0,
            help="Weight for HU Loss (default=0.0=OFF)",
        )
    if "--num_edge_blocks" not in existing:
        parser.add_argument(
            "--num_edge_blocks", type=int, default=2,
            help="Number of EdgeDilatedResidualBlocks at bottleneck (default=2)",
        )
    if "--use_sobel_input" not in existing:
        parser.add_argument(
            "--use_sobel_input", action="store_true", default=True,
            help="Use FixedSobelLayer at bottleneck (default=True)",
        )
        parser.add_argument(
            "--no_sobel_input", action="store_false", dest="use_sobel_input",
            help="Disable FixedSobelLayer for ablation (Variant B)",
        )

    return parser
