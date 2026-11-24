"""Command-line entry point (pipeline under construction)."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Read an analog gauge from an image.")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--min", type=float, default=0.0, help="Scale minimum value")
    parser.add_argument("--max", type=float, default=10.0, help="Scale maximum value")
    args = parser.parse_args()
    raise SystemExit(
        f"Pipeline not implemented yet (image={args.image}). See the README roadmap."
    )


if __name__ == "__main__":
    main()
