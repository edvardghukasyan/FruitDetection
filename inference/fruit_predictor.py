import sys
import os
import json
import torch
from skimage import io

root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.abspath(root))
from training import FruitDetector, get_network
from preprocessing import process_image


class FruitPredictor:
    @staticmethod
    def __read_config():
        with open(os.path.join(root, "config.json"), "r") as config:
            return json.load(config)

    def __load_model(self, num_classes):
        checkpoint_path = os.path.join(root, 'model', 'fruit_detection_model.ckpt')
        # Get the network
        network = get_network(num_classes)
        # Load checkpoint into model
        self.model = FruitDetector.load_from_checkpoint(checkpoint_path, num_classes=num_classes, network=network)
        # Set the model to evaluation mode
        self.model.eval()

    def __init__(self):
        config = FruitPredictor.__read_config()
        # Store image size for preprocessing
        self.image_size = config["preprocess_config"]['image_size']
        # Extract number of classes from config file
        num_classes = config["train_config"]['num_classes']
        # Load model and set model attribute
        self.__load_model(num_classes)

    def __process_image(self, image_path):
        image = process_image(io.imread(image_path), image_size=self.image_size)
        torch_image = torch.from_numpy(image)
        # Torch requires float and other permutation of dimensions
        torch_image = torch.permute(torch_image, (2, 0, 1)).float()
        # Torch also requires batch size, so add one additional dimension
        return torch.unsqueeze(torch_image, 0)

    def __model_outputs(self, image_path):
        image = self.__process_image(image_path)
        # Make predictions
        with torch.no_grad():
            return self.model.network(image)

    def predict(self, image_path):
        outputs = self.__model_outputs(image_path)
        _, predicted = torch.max(outputs, 1)

        return predicted


fruit_predictor = FruitPredictor()
print('Prediction:', fruit_predictor.predict('../fruits360_processed/Test/apple/6r0_3.jpg'))
