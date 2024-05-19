import sys
import os
import torch
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from training import FruitDetector, get_network

# Get the Conda environment root directory
root = os.path.dirname(os.path.dirname(__file__))
checkpoint_path = os.path.join(root, 'model', 'fruit_detection_model.ckpt')

model = FruitDetector.load_from_checkpoint(checkpoint_path, num_classes=7, network=get_network(7))
model.eval()

# Define the image transformations
preprocess = transforms.Compose([
    transforms.ToTensor()
])

# Load and preprocess the image
image_path = '../fruits360_processed/Test/apple/6r0_3.jpg'
image = preprocess(Image.open(image_path)).unsqueeze(0)

# Make predictions
with torch.no_grad():
    outputs = model.network(image)
    _, predicted = torch.max(outputs, 1)
    print(outputs)

print(f'Predicted class: {predicted.item()}')
