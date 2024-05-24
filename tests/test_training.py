import os
import pytest
import shutil
from pathlib import Path
from datetime import datetime
import json
from training.fruit_train import train



@pytest.mark.skip(reason="This test will train the model, which may take a long time.")
def test_train_model():
    # Load the configuration
    config_path = os.path.join(os.path.dirname(__file__), "../config.json")
    with open(config_path, "r") as config_file:
        train_config = json.load(config_file)["train_config"]

    # Ensure the checkpoints and logs directories are cleaned up
    checkpoints_dir = os.path.join(os.path.dirname(__file__), "../checkpoints")
    logs_dir = os.path.join(os.path.dirname(__file__), "../logs")
    if os.path.exists(checkpoints_dir):
        shutil.rmtree(checkpoints_dir)
    if os.path.exists(logs_dir):
        shutil.rmtree(logs_dir)

    # Run the training process
    train(**train_config)

    # Check if the checkpoints directory has been created and contains files
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_checkpoints_dir = os.path.join(checkpoints_dir, now)
    assert os.path.isdir(current_checkpoints_dir), "Checkpoints directory not found after training."

    # Check if any checkpoint files are present
    checkpoint_files = list(Path(current_checkpoints_dir).glob("*.ckpt"))
    assert len(checkpoint_files) > 0, "No checkpoint files found after training."

    # Clean up (optional)
    if os.path.exists(checkpoints_dir):
        shutil.rmtree(checkpoints_dir)
    if os.path.exists(logs_dir):
        shutil.rmtree(logs_dir)


if __name__ == "__main__":
    pytest.main(["-s", "test_training.py"])