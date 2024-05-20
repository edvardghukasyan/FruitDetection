import sys
import os
import torch
import pytest
import random
from tests.utils import root, preprocess_config, fruits

sys.path.append(os.path.abspath(root))
from inference import FruitPredictor


@pytest.mark.medium
@pytest.mark.parametrize('image_path', [
    os.path.join(root, 'fruits360_merged', 'Test', 'eggplant', 'violet_1r0_3.jpg'),
    os.path.join(root, 'fruits360_merged', 'Training', 'cucumber', '1r0_16.jpg'),
    os.path.join(root, 'fruits360_merged', 'Validation', 'carrot', '1r0_193.jpg'),
    pytest.param(os.path.join(root, 'fruits360_merged', 'Test', 'cabbage', 'wrong_path.jpg'), marks=pytest.mark.xfail)
])
def test_fruit_predictor_preprocessing(image_path):
    fruit_predictor = FruitPredictor()
    image_tensor = fruit_predictor._FruitPredictor__preprocess_image(image_path)
    image_size = preprocess_config['image_size']
    assert image_tensor.shape == (1, 3, image_size, image_size)


@pytest.mark.medium
@pytest.mark.parametrize('image_path', [
    os.path.join(root, 'fruits360_merged', 'Test', 'eggplant', 'violet_1r0_3.jpg'),
    os.path.join(root, 'fruits360_merged', 'Training', 'cucumber', '1r0_16.jpg'),
    os.path.join(root, 'fruits360_merged', 'Validation', 'carrot', '1r0_193.jpg'),
    pytest.param(os.path.join(root, 'fruits360_merged', 'Test', 'cabbage', 'wrong_path.jpg'), marks=pytest.mark.xfail)
])
def test_fruit_predict(image_path):
    fruit_predictor = FruitPredictor()
    assert fruit_predictor.predict_from_path(image_path) in fruits


@pytest.mark.medium
@pytest.mark.parametrize('image_path', [
    os.path.join(root, 'fruits360_merged', 'Test', 'eggplant', 'violet_1r0_3.jpg'),
    os.path.join(root, 'fruits360_merged', 'Training', 'cucumber', '1r0_16.jpg'),
    os.path.join(root, 'fruits360_merged', 'Validation', 'carrot', '1r0_193.jpg'),
    pytest.param(os.path.join(root, 'fruits360_merged', 'Test', 'cabbage', 'wrong_path.jpg'), marks=pytest.mark.xfail)
])
def test_fruit_predict_same_inputs(image_path):
    fruit_predictor = FruitPredictor()
    predictions = fruit_predictor.predict_from_paths([image_path, image_path, image_path])
    for prediction in predictions:
        assert prediction in fruits
        assert prediction == predictions[0]


@pytest.mark.medium
def test_fruit_predict_multiple_inputs():
    image_paths = [
        os.path.join(root, 'fruits360_merged', 'Test', 'eggplant', 'violet_1r0_3.jpg'),
        os.path.join(root, 'fruits360_merged', 'Training', 'cucumber', '1r0_16.jpg'),
        os.path.join(root, 'fruits360_merged', 'Validation', 'carrot', '1r0_193.jpg')
    ]
    fruit_predictor = FruitPredictor()
    predictions = fruit_predictor.predict_from_paths(image_paths)
    for prediction in predictions:
        assert prediction in fruits


@pytest.mark.medium
def test_fruit_predict_random_tensor():
    fruit_predictor = FruitPredictor()
    image_size = preprocess_config['image_size']
    batch_size = 5
    image_tensor = torch.randn(batch_size, 3, image_size, image_size)
    predictions = fruit_predictor.predict_from_tensor(image_tensor)
    assert len(predictions) == batch_size
    for prediction in predictions:
        assert prediction in fruits


@pytest.mark.medium
def test_fruit_predict_bad_input():
    fruit_predictor = FruitPredictor()
    image_size = preprocess_config['image_size']
    batch_size = 5

    with pytest.raises(RuntimeError):
        image_tensor = torch.randn(batch_size, 10, image_size, image_size)
        fruit_predictor.predict_from_tensor(image_tensor)


@pytest.fixture(params=[
    os.path.join(root, 'fruits360_merged', phase, fruit)
    for phase in ['Test', 'Training', 'Validation']
    for fruit in fruits
])
def directory_image_files(request):
    data_dir = request.param
    image_paths = []
    for _, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(data_dir, file)
                image_paths.append(image_path)
    sample_size = 30
    return random.sample(image_paths, sample_size)


@pytest.mark.large
def test_fruit_predictions_for_the_directory(directory_image_files):
    fruit_predictor = FruitPredictor()
    predictions = fruit_predictor.predict_from_paths(directory_image_files)
    for prediction in predictions:
        assert prediction in fruits
