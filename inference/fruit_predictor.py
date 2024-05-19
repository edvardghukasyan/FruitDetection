import sys
import os
import json
import torch
import pickle
from skimage import io

root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.abspath(root))
from training import FruitDetector, get_network
from preprocessing import process_image


class FruitPredictor:
    def __init__(self):
        config = FruitPredictor.__read_config()
        # Store image size for preprocessing
        self.image_size = config["preprocess_config"]['image_size']
        # Extract number of classes from config file
        num_classes = config["train_config"]['num_classes']
        # Load model and set model attribute
        self.__load_model(num_classes)
        # Load and create index to fruit mapping
        self.__load_index_to_fruit_mapping()

    def predict_from_paths(self, image_paths):
        """
        Predicts fruits in the images determined by passed image_paths list
        :param image_paths: Iterable object consisting valid image paths
        :return: predicted fruits list
        """
        outputs = self.__model_outputs_from_paths(image_paths)
        _, predicted = torch.max(outputs, 1)

        return self.__index_to_fruit(predicted)

    def predict_from_path(self, image_path):
        """
        Predicts fruits in the images determined by passed image_paths list
        :param image_path: Valid image path
        :return: predicted fruit
        """
        return self.predict_from_paths([image_path])[0]

    def predict_from_tensor(self, tensor):
        """
        Predicts fruits in the images determined by passed images tensor
        :param tensor: Expects 4-dimensional tensor with following structure: (batch_size, channels, width, height)
        :return: predicted fruits list
        """
        outputs = self.__model_outputs_from_tensor(tensor)
        _, predicted = torch.max(outputs, 1)

        return self.__index_to_fruit(predicted)

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

    def __load_index_to_fruit_mapping(self):
        mapping_path = os.path.join(root, 'model', 'fruit_to_index.pkl')
        with open(mapping_path, 'rb') as file:
            # Load fruit to index mapping
            reverse_mapping = pickle.load(file)
            # Create index to fruit mapping
            self.index_to_fruit_mapping = {}
            for k, v in reverse_mapping.items():
                self.index_to_fruit_mapping[v] = k

    def __index_to_fruit(self, tensor):
        return list(map(lambda index: self.index_to_fruit_mapping[int(index)], tensor))

    def __preprocess_image(self, image_path):
        image = process_image(io.imread(image_path), image_size=self.image_size)
        torch_image = torch.from_numpy(image)
        # Torch requires float and other permutation of dimensions
        torch_image = torch.permute(torch_image, (2, 0, 1)).float()
        # Torch also requires batch size, so add one additional dimension
        return torch.unsqueeze(torch_image, 0)

    def __model_outputs_from_tensor(self, batch_tensor):
        # Make predictions
        with torch.no_grad():
            return self.model.network(batch_tensor)

    def __model_outputs_from_paths(self, image_paths):
        # Concat image tensors into batch tensor
        batch_tensor = torch.cat([self.__preprocess_image(path) for path in image_paths], dim=0)
        return self.__model_outputs_from_tensor(batch_tensor)
