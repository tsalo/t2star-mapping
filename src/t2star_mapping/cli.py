from __future__ import annotations

import argparse
import os

from .pipeline import PipelineOptions, T2StarPipeline


def main() -> None:
    p = argparse.ArgumentParser(
        description="T2* mapping with through-slice dropout correction",
    )
    p.add_argument(
        "--magn",
        required=True,
        help="Path to multi-echo magnitude 4D NIfTI",
    )
    p.add_argument(
        "--phase",
        required=True,
        help="Path to multi-echo phase 4D NIfTI",
    )
    p.add_argument(
        "--te",
        required=True,
        nargs="+",
        type=float,
        help="Echo times in ms (space-separated)",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output directory",
    )
    p.add_argument(
        "--prefix",
        default="",
        help="Output filename prefix",
    )
    p.add_argument(
        "--method",
        default="nlls",
        choices=["ols", "gls", "nlls", "num"],
        help="Fitting method",
    )
    p.add_argument(
        "--rmse-thresh",
        default=0.8,
        type=float,
        help="RMSE threshold for frequency fit mask",
    )
    p.add_argument(
        "--poly-order",
        default=3,
        type=int,
        help="3D polynomial order for smoothing",
    )
    p.add_argument(
        "--downsample",
        default=(2, 2, 2),
        type=int,
        nargs=3,
        help="Downsample factors x y z",
    )
    p.add_argument(
        "--dz-mm",
        default=1.25,
        type=float,
        help="Slice thickness in mm",
    )
    p.add_argument(
        "--t2max",
        default=1000.0,
        type=float,
        help="Max T2* to clamp (ms)",
    )
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    opts = PipelineOptions(
        prefix=args.prefix,
        fitting_method=args.method,
        echo_times_ms=args.te,
        rmse_thresh=args.rmse_thresh,
        smooth_poly_order=args.poly_order,
        downsample=tuple(args.downsample),
        dz_mm=args.dz_mm,
        threshold_t2star_max_ms=args.t2max,
    )
    pipe = T2StarPipeline(args.magn, args.phase, opts)
    paths = pipe.run(args.out)
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
