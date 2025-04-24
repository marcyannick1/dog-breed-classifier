import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Préparer les générateurs d'images
train_datagen = ImageDataGenerator(rescale=1./255)  # Normalisation des pixels (entre 0 et 1)
test_datagen = ImageDataGenerator(rescale=1./255)

# Chargement des images d'entraînement
train_generator = train_datagen.flow_from_directory(
    'Stanford_Dogs_Dataset/train',  # Répertoire contenant les sous-dossiers des races
    target_size=(224, 224),  # Taille des images d'entrée pour le modèle
    batch_size=32,  # Nombre d'images par lot
    class_mode='categorical',  # Car on a plusieurs classes (races)
)

# Chargement des images de test
test_generator = test_datagen.flow_from_directory(
    'Stanford_Dogs_Dataset/test',  # Répertoire contenant les sous-dossiers des races
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
)

# Construire le modèle CNN
model = Sequential()

# Première couche convolutive + max pooling
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))

# Deuxième couche convolutive + max pooling
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Troisième couche convolutive + max pooling
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Aplatir les images 2D pour les passer dans des couches denses
model.add(Flatten())

# Couche dense pour les prédictions
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout pour éviter le sur-apprentissage

# Couche de sortie : un neurone par classe (race)
model.add(Dense(train_generator.num_classes, activation='softmax'))

# Compiler le modèle
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,  # Nombre d'époques (tu peux augmenter si nécessaire)
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size,
)

# Évaluation sur le jeu de test
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Sauvegarder le modèle
model.save('dog_breed_classifier.h5')