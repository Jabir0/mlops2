from fastapi import FastAPI
import tensorflow as tf
import numpy as np
import uvicorn
import os

app = FastAPI()

# Load model
MODEL_DIR = "saved_model"
model = tf.keras.models.load_model(MODEL_DIR)

@app.get("/")
def home():
    return {"message": "Heart Disease Detection Model is running on Heroku!"}

@app.post("/predict")
def predict(data: dict):
    try:
        input_data = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(input_data)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
