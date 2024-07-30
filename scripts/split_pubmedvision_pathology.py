import os
import json


if __name__ == "__main__":
    file_path = "../cache/PubMedVision_Alignment_VQA.json"

    with open(file_path, "r") as f:
        data = json.load(f)

    new_data = []
    for sample in data:
        if sample["modality"] == "Digital Photography":
            new_data.append(sample)
        # break

    with open("../cache/PubMedVision_Alignment_PATH_VQA.json", "W") as f:
        json.dump(new_data, f, indent=2)