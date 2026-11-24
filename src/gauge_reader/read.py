"""Command-line entry point: read one gauge image.

Example:
    python -m gauge_reader.read --image gauge.jpg --min-angle 225 --max-angle 135 \
        --min 0 --max 10 --debug out.png
"""

import argparse

import cv2

from .dial import DialSpec
from .reader import draw_reading, read_gauge


def main() -> None:
    p = argparse.ArgumentParser(description="Read an analog gauge from an image.")
    p.add_argument("--image", required=True, help="path to the input image")
    p.add_argument("--min", type=float, default=0.0, help="value at the start of the scale")
    p.add_argument("--max", type=float, default=10.0, help="value at the end of the scale")
    p.add_argument("--min-angle", type=float, default=225.0,
                   help="angle of the scale start, degrees clockwise from 12 o'clock (default 225 = 7:30)")
    p.add_argument("--max-angle", type=float, default=135.0,
                   help="angle of the scale end (default 135 = 4:30)")
    p.add_argument("--debug", help="optional path to save an annotated image")
    args = p.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")
    spec = DialSpec(args.min_angle, args.max_angle, args.min, args.max)
    reading = read_gauge(img, spec)
    if reading is None:
        raise SystemExit("No gauge found in the image.")
    print(f"value={reading.value:.3f}  angle={reading.angle_deg:.1f}°  confidence={reading.confidence:.1f}")
    if args.debug:
        cv2.imwrite(args.debug, draw_reading(img, reading))
        print(f"annotated image saved to {args.debug}")


if __name__ == "__main__":
    main()
