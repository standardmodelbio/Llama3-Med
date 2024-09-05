import argparse
import json
import os

from loguru import logger
from PIL import Image
from tqdm import tqdm


def process_file(modality, input_file, output_file, image_dir):
    with open(input_file, "r") as f:
        data = json.load(f)

    new_sample = []
    if not modality:
        for sample in tqdm(data, desc="Processing samples"):
            try:
                for image_file in sample["image"]:
                    Image.open(os.path.join(image_dir, image_file))
                sample["conversations"][0]["value"] = (
                    "<image>\n" * len(sample["image"])
                    + sample["conversations"][0]["value"]
                )
                new_sample.append(sample)
            except Exception as e:
                logger.warning(f"Error processing {sample['image']}: {e}")
    else:
        for sample in tqdm(data, desc="Processing samples"):
            if sample["modality"] == modality:
                try:
                    for image_file in sample["image"]:
                        Image.open(os.path.join(image_dir, image_file))
                    sample["conversations"][0]["value"] = (
                        "<image>\n" * len(sample["image"])
                        + sample["conversations"][0]["value"]
                    )
                    new_sample.append(sample)
                except Exception as e:
                    logger.warning(f"Error processing {sample['image']}: {e}")

    with open(output_file, "w") as f:
        json.dump(new_sample, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Process PubMedVision JSON files.")
    parser.add_argument(
        "--modality",
        default="Digital Photography",
        help="Modality to extract",
    )
    parser.add_argument(
        "--align_input",
        default="../data/pubmedvision/PubMedVision_Alignment_VQA.json",
        help="Input file path for Alignment data",
    )
    parser.add_argument(
        "--align_output",
        default="../data/pubmedvision/PubMedVision_Alignment_PATH_VQA.json",
        help="Output file path for processed Alignment data",
    )
    parser.add_argument(
        "--instruct_input",
        default="../data/pubmedvision/PubMedVision_InstructionTuning_VQA.json",
        help="Input file path for InstructionTuning data",
    )
    parser.add_argument(
        "--instruct_output",
        default="../data/pubmedvision/PubMedVision_InstructionTuning_PATH_VQA.json",
        help="Output file path for processed InstructionTuning data",
    )
    parser.add_argument(
        "--image_dir",
        default="../data/pubmedvision",
        help="Directory containing the image files",
    )

    args = parser.parse_args()

    logger.info("Processing Alignment data...")
    process_file(args.modality, args.align_input, args.align_output, args.image_dir)

    logger.info("Processing InstructionTuning data...")
    process_file(
        args.modality, args.instruct_input, args.instruct_output, args.image_dir
    )

    logger.info("Processing complete.")


if __name__ == "__main__":
    main()
