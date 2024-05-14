import os
import shutil

import click
from tqdm import tqdm


@click.command()
@click.option("--input_dir", default="../fruits360/fruits-360-original-size/fruits-360-original-size")
@click.option("--output_dir", default="../fruits360_merged")
def merge(
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
    merge()
