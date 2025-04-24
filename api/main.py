import os
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO

# Charger le modèle
model = tf.keras.models.load_model('model/dog_breed_classifier_mobileNet.h5')

# Récupérer les noms des classes, triés pour correspondre à flow_from_directory
train_dir = '../ia/Stanford_Dogs_Dataset/train'
class_names = sorted(os.listdir(train_dir))

# Créer l'app FastAPI
app = FastAPI()

# Prétraiter l'image
def preprocess_image(file):
    img_bytes = BytesIO(file.file.read())
    img = image.load_img(img_bytes, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_array = preprocess_image(file)
    predictions = model.predict(img_array)[0]  # [0] pour enlever la batch dimension

    # Obtenir les indices des 3 meilleures prédictions
    top_indices = predictions.argsort()[-3:][::-1]
    top_preds = [
        {"breed": class_names[i], "confidence": round(float(predictions[i]) * 100, 2)}
        for i in top_indices
    ]

    return JSONResponse(content={"top_predictions": top_preds})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)