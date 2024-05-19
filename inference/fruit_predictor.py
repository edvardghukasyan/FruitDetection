import sys
import os
import json
import torch
import torch.nn.functional as F
from skimage import io

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from training import FruitDetector, get_network
from preprocessing import process_image


class FruitPredictor:
    def __init__(self):
        # Get the Conda environment root directory
        root = os.path.dirname(os.path.dirname(__file__))
        checkpoint_path = os.path.join(root, 'model', 'fruit_detection_model.ckpt')
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json"), "r") as config:
            # Extract number of classes from config file
            config = json.load(config)
            num_classes = config["train_config"]['num_classes']
            # Get the network
            network = get_network(num_classes)
            # Load checkpoint into model
            self.model = FruitDetector.load_from_checkpoint(checkpoint_path, num_classes=num_classes, network=network)
            self.image_size = config["preprocess_config"]['image_size']
        # Set the model to evaluation mode
        self.model.eval()

    def __read_image_tensor(self, image_path):
        image = process_image(io.imread(image_path), image_size=self.image_size)
        torch_image = torch.from_numpy(image)
        # Torch requires float and other permutation of dimensions
        return torch.permute(torch_image, (2, 0, 1)).float()

    def predict_proba(self, image_path):
        image = self.__read_image_tensor(image_path)
        image = torch.unsqueeze(image, 0)

        # Make predictions
        with torch.no_grad():
            return self.model.network(image)

    def predict(self, image_path):
        outputs = self.predict_proba(image_path)
        _, predicted = torch.max(outputs, 1)

        return predicted


fruit_predictor = FruitPredictor()
print('Prediction:', fruit_predictor.predict('../fruits360_processed/Test/apple/6r0_3.jpg'))
