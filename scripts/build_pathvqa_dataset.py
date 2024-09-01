import argparse
import json

from tqdm import tqdm
from loguru import logger


def process_data(input_file, output_file):
    with open(input_file, "r") as f:
        data = json.load(f)

    with open(output_file, "w") as ans_file:
        for sample in tqdm(data, desc="Processing samples"):
            temp = {
                "question_id": sample["id"],
                "image": sample["image"],
                "text": sample["conversations"][0]["value"],
                "answer_type": sample["answer_type"],
            }
            ans_file.write(json.dumps(temp) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Process VQA JSON data and convert to JSONL format."
    )
    parser.add_argument(
        "--input",
        default="../data/vqa_path/test_llava.json",
        help="Input JSON file path",
    )
    parser.add_argument(
        "--output",
        default="../data/vqa_path/pathvqa_test_questions.jsonl",
        help="Output JSONL file path",
    )

    args = parser.parse_args()

    logger.info(f"Processing data from {args.input}")
    process_data(args.input, args.output)
    logger.info(f"Processing complete. Output saved to {args.output}")


if __name__ == "__main__":
    main()
