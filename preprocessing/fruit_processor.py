import json
import os

import imageio
import numpy as np
from skimage import io
from skimage.transform import resize
from tqdm import tqdm


def pad_to_square(image: np.ndarray, fill_value: int = 255):
    img_size = max(image.shape)
    if img_size > image.shape[0]:
        pad_size = (img_size - image.shape[0]) // 2
        pad = fill_value * np.ones(shape=(pad_size, img_size, *image.shape[2:]), dtype=np.uint8)
        image = np.concatenate((pad, image, pad), axis=0)
    
    elif img_size > image.shape[1]:
        pad_size = (img_size - image.shape[1]) // 2
        pad = fill_value * np.ones(shape=(img_size, pad_size, *image.shape[2:]), dtype=np.uint8)
        image = np.concatenate((pad, image, pad), axis=1)
    
    return image


def process_image(image: np.ndarray, image_size: int):
    image = pad_to_square(image)
    image = (resize(image, (image_size, image_size, *image.shape[2:])) * 255).astype(np.uint8)
    return image


def process_fruits(
    input_dir: str,
    output_dir: str,
    image_size: int
):
    for split in ["Training", "Validation", "Test"]:
        print(split)
        split_path = os.path.join(input_dir, split)
        for c in tqdm(os.listdir(split_path)):
            class_path = os.path.join(split_path, c)
            output_path = os.path.join(output_dir, split, c)
            os.makedirs(output_path, exist_ok=True)
            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)
                image = io.imread(file_path)
                image = process_image(image, image_size=image_size)
                imageio.imwrite(os.path.join(output_path, file), image)


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json"), "r") as config:
        preprocess_config = json.load(config)["preprocess_config"]
        process_fruits(**preprocess_config)
