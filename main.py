from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from inference.fruit_predictor import FruitPredictor
import io
from PIL import Image
from typing import List

# Description with local image reference using HTML
description = """
API for detecting fruits from images using a pre-trained model.
The API provides two endpoints: one for retrieving model information and another for making predictions.

## Model Information

- **Algorithm**: Convolutional Neural Network (CNN)
- **Version**: 1.2
- **Training Date**: 2024-05-10
- **Dataset**: [Fruits-360 Dataset](https://www.kaggle.com/datasets/moltean/fruits)
- **Dataset Paper**: [Fruits-360 dataset: new research directions](https://www.researchgate.net/publication/354535752_Fruits-360_dataset_new_research_directions)
- **Related Research**: [Fruit recognition from images using deep learning](https://arxiv.org/abs/1712.00580)
- **ResNet Paper**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

## Output Labels
- ![apple](https://via.placeholder.com/20/FF0000/FFFFFF?text=A) apple
- ![cabbage](https://via.placeholder.com/20/008000/FFFFFF?text=C) cabbage
- ![carrot](https://via.placeholder.com/20/FFA500/FFFFFF?text=C) carrot
- ![cucumber](https://via.placeholder.com/20/008000/FFFFFF?text=C) cucumber
- ![eggplant](https://via.placeholder.com/20/800080/FFFFFF?text=E) eggplant
- ![pear](https://via.placeholder.com/20/FFFF00/000000?text=P) pear
- ![zucchini](https://via.placeholder.com/20/008000/FFFFFF?text=Z) zucchini

## Example Image
<img src="http://127.0.0.1:8000/static/kargin.jpeg" alt="Example Image" width="500">
"""

app = FastAPI(
    title="Fruit Detection API",
    description=description,
    version="1.0"
)

# Serve static files from the "static" directory
app.mount("/static", StaticFiles(directory="static"), name="static")

fruit_predictor = FruitPredictor()

class ModelInfo(BaseModel):
    algorithm: str
    version: str
    training_date: str
    dataset: str
    dataset_paper: str
    related_research: str
    resnet_paper: str
    output_labels: List[str]

@app.get("/model_info", response_model=ModelInfo, summary="Get Model Information",
         description="Retrieve detailed information about the machine learning model used for fruit detection.")
def get_model_info():
    return {
        "algorithm": "Convolutional Neural Network",
        "version": "1.0",
        "training_date": "2024-05-20",
        "dataset": "https://www.kaggle.com/datasets/moltean/fruits",
        "dataset_paper": "https://www.researchgate.net/publication/354535752_Fruits-360_dataset_new_research_directions",
        "related_research": "https://arxiv.org/abs/1712.00580",
        "resnet_paper": "https://arxiv.org/abs/1512.03385",
        "output_labels": ['apple', 'cabbage', 'carrot', 'cucumber', 'eggplant', 'pear', 'zucchini']
    }

@app.post("/predict", summary="Predict Fruit from Image",
          description="Upload an image to get a fruit prediction from the trained model. The model processes the input image and returns the predicted fruit label.")
async def predict(file: UploadFile = File(...)):
    """
    Upload an image file and get the predicted fruit label from the trained model.

    Parameters:
    - file: UploadFile (an image file to be processed)

    Returns:
    - JSON object with the predicted fruit label
    """
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    temp_image_path = "temp_image.jpg"
    image.save(temp_image_path)

    predicted_fruit = fruit_predictor.predict_from_path(temp_image_path)

    return {"prediction": predicted_fruit}

@app.post("/predict_multiple", summary="Predict Fruits from Multiple Images",
          description="Upload multiple images to get fruit predictions from the trained model. The model processes the input images and returns the predicted fruit labels.")
async def predict_multiple(files: List[UploadFile] = File(...)):
    """
    Upload multiple image files and get the predicted fruit labels from the trained model.

    Parameters:
    - files: List[UploadFile] (image files to be processed)

    Returns:
    - JSON object with the predicted fruit labels
    """
    image_paths = []
    for file in files:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        temp_image_path = f"temp_{file.filename}"
        image.save(temp_image_path)
        image_paths.append(temp_image_path)

    predicted_fruits = fruit_predictor.predict_from_paths(image_paths)

    return {"predictions": predicted_fruits}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)