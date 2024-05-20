import os
import sys
import pytest
from tests.utils import root, fruits


@pytest.mark.parametrize("dir_name", [
    'fruits360_merged',
    'fruits360_processed'
])
@pytest.mark.parametrize("phase", [
    'Test',
    'Training',
    'Validation',
    pytest.param('Verification', marks=pytest.mark.xfail)
])
@pytest.mark.parametrize("fruit", fruits)
def test_data_existence(dir_name, phase, fruit):
    directory = os.path.join(root, dir_name, phase, fruit)
    assert os.path.isdir(directory)
