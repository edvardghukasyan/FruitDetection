import os
import pytest


@pytest.mark.parametrize("dir_name", [
    'fruits360_merged',
    'fruits360_processed'
])
def test_required_directory_existence(dir_name):
    root = os.path.dirname(os.path.dirname(__file__))
    directory = os.path.join(root, dir_name)
    assert os.path.isdir(directory)
