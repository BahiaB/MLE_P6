# app.py
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from pyngrok import ngrok
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
from fastapi.responses import JSONResponse
from joblib import load
import os
print("Libraries loaded")
ngrok.set_auth_token("2jcEz2RcEwdeOBJiX6iYEo9EAUC_3PrykTqQzCMdzpVxfWEq4")
os.environ["NGROK_AUTH_TOKEN"] = "2jcEz2RcEwdeOBJiX6iYEo9EAUC_3PrykTqQzCMdzpVxfWEq4"



app = FastAPI()

def image_prep_fct(data):
    
    image = Image.open(BytesIO(data))
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    prepared_image_np = np.array(image)
    # Add a batch dimension to match the model's expected input shape
    prepared_image_np = np.expand_dims(prepared_image_np, axis=0)
    return prepared_image_np

try:
    model = tf.keras.models.load_model("/content/drive/Othercomputers/Mon_MacBook_Air/MLE_P6/model2_best_weights.keras")
except Exception as e:
    print(f"Failed to load models: {e}")
    raise e

@app.get("/")
def read_root():
    return {"Hello": "World"}
class Item(BaseModel):
    text: str

@app.post("/predict/")
async def make_prediction(file: UploadFile = File(...)):
    try:
        # Read image file as PIL Image
        image = await file.read()
        preprocessed_image = image_prep_fct(image)
        # Preprocess the image - Example: Resize and scale
        #image = image.resize((224, 224))  # Assuming model expects 224x224 images
        #image_array = np.array(image) / 255.0  # Scale pixel values to [0, 1]
        #image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        #image_array = preprocess_input(image_array)  # Preprocess image as expected by VGG16

        # Make prediction
        prediction = model.predict(preprocessed_image)
        # Convert prediction to desired format/response
        print(prediction)
        prediction_result = np.argmax(prediction, axis=1)  # Example: Get class with highest probability
        print(prediction_result)
        if int(prediction_result[0]) == 2:
            return {"prediction": "boxer"}
        elif int(prediction_result[0]) == 1:
            return {"prediction": "eskimo dog"}
        elif int(prediction_result[0]) == 0:
            return {"prediction": "chihuahua"}
        else:
            return {"prediction": int(prediction_result[0])}
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"Error processing image: {e}"})


# Configure ngrok
ngrok.set_auth_token("2jcEz2RcEwdeOBJiX6iYEo9EAUC_3PrykTqQzCMdzpVxfWEq4")
public_url = ngrok.connect(8000, bind_tls=True)
print("ngrok tunnel 'public_url':", public_url)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)