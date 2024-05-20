import pickle
import os
import json

root = os.path.dirname(os.path.dirname(__file__))


def load_fruit_to_index_mapping():
    mapping_path = os.path.join(root, 'model', 'fruit_to_index.pkl')
    with open(mapping_path, 'rb') as file:
        # Load fruit to index mapping
        return pickle.load(file)


def load_config():
    with open(os.path.join(root, "config.json"), "r") as config:
        return json.load(config)


fruits = load_fruit_to_index_mapping().keys()

config = load_config()
train_config = config['train_config']
preprocess_config = config['preprocess_config']
data_merge_config = config['data_merge_config']
