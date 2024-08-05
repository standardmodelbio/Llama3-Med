import json
import os

from PIL import Image
from tqdm import tqdm

if __name__ == "__main__":
    # Alignment
    file_path = "../cache/pubmedvision/PubMedVision_Alignment_VQA.json"

    with open(file_path, "r") as f:
        data = json.load(f)

    new_sample = []
    for sample in tqdm(data):
        if sample["modality"] == "Digital Photography":
            try:
                for image_file in sample["image"]:
                    image = Image.open(
                        os.path.join("../cache/pubmedvision", image_file)
                    )
                # sample["image"] = sample["image"][0]
                sample["conversations"][0]["value"] = "<image>\n" * len(sample["image"]) + sample["conversations"][0]["value"]
                new_sample.append(sample)
            except:
                print(sample["image"])

    save_file_path = "../cache/pubmedvision/PubMedVision_Alignment_PATH_VQA.json"
    with open(save_file_path, "w") as f:
        json.dump(new_sample, f, indent=2)

    # InstructionTuning
    file_path = "../cache/pubmedvision/PubMedVision_InstructionTuning_VQA.json"

    with open(file_path, "r") as f:
        data = json.load(f)

    new_sample = []
    for sample in tqdm(data):
        if sample["modality"] == "Digital Photography":
            try:
                for image_file in sample["image"]:
                    image = Image.open(
                        os.path.join("../cache/pubmedvision", image_file)
                    )
                # sample["image"] = sample["image"][0]
                sample["conversations"][0]["value"] = "<image>\n" * len(sample["image"]) + sample["conversations"][0]["value"]
                new_sample.append(sample)
            except:
                print(sample["image"])

    save_file_path = (
        "../cache/pubmedvision/PubMedVision_InstructionTuning_PATH_VQA.json"
    )
    with open(save_file_path, "w") as f:
        json.dump(new_sample, f, indent=2)
