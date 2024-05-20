import os
import sys
import pytest
import numpy as np
from skimage import io
from tests.utils import root, fruits, preprocess_config

sys.path.append(os.path.abspath(root))
from preprocessing import process_image


@pytest.mark.parametrize('dir_name', [
    'fruits360_merged',
    'fruits360_processed'
])
@pytest.mark.parametrize('phase', [
    'Test',
    'Training',
    'Validation',
    pytest.param('Verification', marks=pytest.mark.xfail)
])
@pytest.mark.parametrize('fruit', fruits)
def test_data_existence(dir_name, phase, fruit):
    directory = os.path.join(root, dir_name, phase, fruit)
    assert os.path.isdir(directory)


@pytest.fixture(params=[
    (100, 40),
    (1000, 300),
    (50, 50)
])
def random_numpy_image_input(request):
    return np.random.randn(*request.param, 3)


@pytest.mark.parametrize('image_size', [
    preprocess_config['image_size'],
    100
])
def test_process_image(random_numpy_image_input, image_size):
    processed_image = process_image(random_numpy_image_input, image_size)
    assert processed_image.shape == (image_size, image_size, 3)


@pytest.fixture(params=[
    os.path.join(root, 'fruits360_merged', 'Test', 'eggplant', 'violet_1r0_3.jpg'),
    os.path.join(root, 'fruits360_merged', 'Training', 'cucumber', '1r0_16.jpg'),
    os.path.join(root, 'fruits360_merged', 'Validation', 'carrot', '1r0_193.jpg'),
    pytest.param(os.path.join(root, 'fruits360_merged', 'Test', 'cabbage', 'wrong_path.jpg'), marks=pytest.mark.xfail)
])
def ndarray_from_image_path(request):
    return io.imread(request.param)


@pytest.mark.parametrize('image_size', [
    preprocess_config['image_size'],
    100
])
def test_process_image_from_path(ndarray_from_image_path, image_size):
    processed_image = process_image(ndarray_from_image_path, image_size)
    assert processed_image.shape == (image_size, image_size, 3)
