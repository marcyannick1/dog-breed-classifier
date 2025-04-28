# 🐶 Dog Breed Classifier

Projet de classification automatique de races de chiens basé sur un modèle de Deep Learning.

## 🔥 Fonctionnalités

- **Modèles IA** :
  - CNN **from scratch**.
  - CNN basé sur **MobileNetV2** (pré-entraîné sur ImageNet).
- **API Backend** :
  - Serveur FastAPI exposant une route de prédiction (`/predict`).
- **Frontend Web** :
  - Interface développée avec **Vite.js** pour uploader une image et afficher la prédiction.

---

## 📂 Structure du projet

```
dog-breed-classifier/
│
├── api/                  # Backend API (FastAPI ou Flask)
│   ├── model/             # Modèles sauvegardés (.h5)
│   ├── main.py            # Serveur principal
│   ├── utils.py           # Utilitaires pour traitement d'image
│   └── requirements.txt   # Dépendances backend
│
├── front/                 # Frontend (Vite.js)
│   ├── public/            # Ressources statiques
│   ├── src/               # Code source (components, pages, etc)
│   └── index.html         # Point d'entrée de l'app
│
├── ia/                    # Scripts IA pour entraîner les modèles
│   ├── Stanford_Dogs_Dataset/  # Dataset utilisé
│   ├── model.py           # CNN from scratch
│   ├── mobilenet.py       # MobileNetV2
│   ├── prepare_dataset.py # Scripts de préparation des données
│   └── requirements.txt   # Dépendances IA
│
└── README.md              # Documentation du projet
```

## 🎓 Dataset

Le projet utilise le **Stanford Dogs Dataset** :

- **Contenu** : 20.580 images de chiens appartenant à 120 races différentes du monde entier
- **Source** : [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- **Structure** :
  - Images organisées en sous-dossiers par race
- **Caractéristiques** :
  - Images de taille et qualité variables
  - Multiples angles et poses
  - Variété de contextes et d'arrière-plans
- **Split** : Dans notre implémentation, les données sont divisées en :
  - 80% pour l'entraînement
  - 20% pour les tests

Le script `prepare_dataset.py` prépare les données pour l'entraînement.

## ⚙️ Installation rapide

### Backend

```bash
cd api
python -m venv .venv
source .venv/bin/activate  # sous Windows : .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend

```bash
cd front
npm install
npm run dev
```

## 🧠 Entraînement des modèles

Dans le dossier `ia/` :

```bash
python model.py          # Entraîne le CNN from scratch
python mobilenet.py      # Entraîne le modèle MobileNetV2
```

## 🚀 Utilisation de l'API

Après avoir lancé le backend :
- Endpoint de prédiction : `POST /predict`
- Payload attendu : image envoyée en `multipart/form-data`

Exemple avec curl :

```bash
curl -X POST -F "file=@/chemin/vers/monimage.jpg" http://127.0.0.1:8000/predict
```

Réponse :

```json
{
  "breed": "golden_retriever",
  "confidence": 0.92
}
```

## 🎨 Frontend

- Upload d'une image depuis votre ordinateur.
- Affichage :
  - de la race prédite,
  - du score de confiance.

---

## 📚 Stack utilisée

- **Python** : TensorFlow, Keras, FastAPI ou Flask
- **JavaScript** : Vite.js
- **Frameworks IA** : MobileNetV2, CNN custom
- **Autres outils** : scikit-learn, matplotlib, Pillow