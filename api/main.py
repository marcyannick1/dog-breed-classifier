import os
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
import requests
from PIL import Image

# Charger le modèle
model = tf.keras.models.load_model('model/dog_breed_classifier.h5')

# Récupérer les noms des classes, triés pour correspondre à flow_from_directory
train_dir = '../ia/Stanford_Dogs_Dataset/train'
class_names = sorted(os.listdir(train_dir))

# Créer l'app FastAPI
app = FastAPI()

# Ajouter le middleware CORS pour autoriser les requêtes depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prétraiter l'image depuis un fichier
def preprocess_image(file):
    img_bytes = BytesIO(file.file.read())
    img = image.load_img(img_bytes, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Prétraiter l'image depuis une URL
def preprocess_image_from_url(image_url):
    response = requests.get(image_url)
    
    if response.status_code != 200:
        raise Exception("L'URL fournie est invalide ou l'image n'a pas pu être téléchargée.")
    
    img = Image.open(BytesIO(response.content))
    
    # Convertir en mode RGB si l'image n'est pas déjà en RGB
    img = img.convert('RGB')
    
    img = img.resize((224, 224))  # Redimensionner à la taille requise
    img_array = np.array(img)
    
    # Assurez-vous que l'image est de type float32 avant de normaliser
    img_array = img_array.astype('float32')
    
    # Normaliser l'image
    img_array /= 255.0
    
    # Ajouter la dimension du batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.get("/")
def root():
    return {"message": "API opérationnelle"}

@app.post("/predict")
async def predict(file: UploadFile = None, url: str = Form(None)):
    img_array = None

    # Si un fichier est fourni
    if file:
        img_array = preprocess_image(file)
    
    # Si une URL est fournie
    elif url:
        try:
            img_array = preprocess_image_from_url(url)
        except Exception as e:
            return JSONResponse(content={"error": f"Erreur lors du traitement de l'URL : {str(e)}"}, status_code=400)
    
    # Si aucune image ni URL n'est fournie
    if img_array is None:
        return JSONResponse(content={"error": "Aucune image ou URL fournie"}, status_code=400)
    
    try:
        # Prédiction
        predictions = model.predict(img_array)[0]  # [0] pour enlever la batch dimension
        
        # Obtenir les indices des 3 meilleures prédictions
        top_indices = predictions.argsort()[-3:][::-1]
        top_preds = [
            {"breed": class_names[i], "confidence": round(float(predictions[i]) * 100, 2)}
            for i in top_indices
        ]
        
        return JSONResponse(content={"top_predictions": top_preds})
    
    except Exception as e:
        return JSONResponse(content={"error": f"Erreur lors de la prédiction : {str(e)}"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
