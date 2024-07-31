import json
import os
from tqdm import tqdm
from PIL import Image


if __name__ == "__main__":
    # Alignment
    file_path = "../cache/pubmedvision/PubMedVision_Alignment_VQA.json"

    with open(file_path, "r") as f:
        data = json.load(f)

    new_sample = []
    for sample in tqdm(data):
        if sample["modality"] == "Digital Photography":
            try:
                image = Image.open(os.path.join("../cache/pubmedvision", sample["image"][0]))
                sample["image"] = sample["image"][0]
                # sample["conversations"][0]["value"] = "<image>\n" + sample["conversations"][0]["value"]
                new_sample.append(sample)
            except:
                print(sample["image"][0])

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
                image = Image.open(os.path.join("../cache/pubmedvision", sample["image"][0]))
                sample["image"] = sample["image"][0]
                # sample["conversations"][0]["value"] = "<image>\n" + sample["conversations"][0]["value"]
                new_sample.append(sample)
            except:
                print(sample["image"][0])

    save_file_path = "../cache/pubmedvision/PubMedVision_InstructionTuning_PATH_VQA.json"
    with open(save_file_path, "w") as f:
        json.dump(new_sample, f, indent=2)