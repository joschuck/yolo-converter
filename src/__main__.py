import argparse
from typing import Tuple

from src.coco import convert_coco_json


def triple_float(value: str) -> Tuple[float, float, float]:
    try:
        # Split the string into three integers and convert them
        a, b, c = map(float, value.split(','))
        # Return the tuple of three integers
        return a, b, c
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid triple of integers: %s" % value)


if __name__ == "__main__":
    # create the parser
    parser = argparse.ArgumentParser(
        description="Script to process input data for various tasks"
    )

    # define arguments
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="coco",
        choices=["coco", "labelimg", "voc"],
        help="input format",
    )
    parser.add_argument(
        "-i", "--input-dir", type=str, required=True, help="input directory"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, required=True, help="output directory"
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        default="detection",
        choices=["detection", "segmentation", "panoptic", "keypoints"],
        help="task type",
    )
    parser.add_argument(
        "--cls91to80",
        action="store_true",
        default=False,
        help="converts COCO 80-index (val2014) to 91-index (paper)",
    )
    parser.add_argument('--split', type=triple_float, default=None, help='Split (in case there is just one JSON')

    # parse the arguments
    args = parser.parse_args()

    # arguments conversion
    format = args.format.lower()

    if format == "coco":
        convert_coco_json(
            args.input_dir,  # directory with *.json
            args.output_dir,
            task=args.task,
            cls91to80=args.cls91to80,
            split=args.split,
        )
    else:
        print("Unknown format!")
