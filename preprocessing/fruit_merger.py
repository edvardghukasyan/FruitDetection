import os
import shutil

import json
from tqdm import tqdm


def merge_fruits(
        input_dir: str,
        output_dir: str
):
    for split in ["Training", "Validation", "Test"]:
        print(split)
        split_path = os.path.join(input_dir, split)
        for c in tqdm(os.listdir(split_path)):
            class_path = os.path.join(split_path, c)
            # class and subclass
            c, sub_c = c.split("_", maxsplit=1)
            output_path = os.path.join(output_dir, split, c)
            os.makedirs(output_path, exist_ok=True)
            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)
                shutil.copyfile(file_path, os.path.join(output_path, sub_c + file))


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json"), "r") as config:
        data_merge_config = json.load(config)["data_merge_config"]
        merge_fruits(**data_merge_config)
