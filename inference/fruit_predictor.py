import sys
import os
import json
import torch
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from training import FruitDetector, get_network


class FruitPredictor:
    def __init__(self):
        # Get the Conda environment root directory
        root = os.path.dirname(os.path.dirname(__file__))
        checkpoint_path = os.path.join(root, 'model', 'fruit_detection_model.ckpt')
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json"), "r") as config:
            # Extract number of classes from config file
            config = json.load(config)["train_config"]
            num_classes = config['num_classes']
            # Get the network
            network = get_network(num_classes)
            # Load checkpoint into model
            self.model = FruitDetector.load_from_checkpoint(checkpoint_path, num_classes=num_classes, network=network)
        # Set the model to evaluation mode
        self.model.eval()

    def predict_proba(self, image_path):
        # Define the image transformations
        preprocess = transforms.Compose([
            transforms.ToTensor()
        ])

        # Load and preprocess the image
        image = preprocess(Image.open(image_path)).unsqueeze(0)

        # Make predictions
        with torch.no_grad():
            return self.model.network(image)

    def predict(self, image_path):
        outputs = self.predict_proba(image_path)
        _, predicted = torch.max(outputs, 1)
        print(predicted)

        return predicted


fruit_predictor = FruitPredictor()
print('Prediction:', fruit_predictor.predict('../fruits360_processed/Test/apple/6r0_3.jpg'))
