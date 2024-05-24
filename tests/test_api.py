import pytest
from fastapi.testclient import TestClient
from main import app
from pathlib import Path

client = TestClient(app)

def test_get_model_info():
    response = client.get("/model_info")
    assert response.status_code == 200
    data = response.json()
    assert data["algorithm"] == "Convolutional Neural Network"
    assert data["version"] == "1.0"
    assert data["training_date"] == "2024-05-20"
    assert data["dataset"] == "https://www.kaggle.com/datasets/moltean/fruits"
    assert data["dataset_paper"] == "https://www.researchgate.net/publication/354535752_Fruits-360_dataset_new_research_directions"
    assert data["related_research"] == "https://arxiv.org/abs/1712.00580"
    assert data["resnet_paper"] == "https://arxiv.org/abs/1512.03385"
    assert data["output_labels"] == ['apple', 'cabbage', 'carrot', 'cucumber', 'eggplant', 'pear', 'zucchini']

def test_predict():
    # Update the path to the sample image
    image_path = Path.cwd().resolve() / 'fruits360_merged/Test/apple/6r0_7.jpg'
    with open(image_path, "rb") as image_file:
        response = client.post("/predict", files={"file": image_file})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
