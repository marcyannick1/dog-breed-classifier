import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

# Chemins
train_dir = "Stanford_Dogs_Dataset/train"
test_dir = "Stanford_Dogs_Dataset/test"

# Générateurs d'images avec data augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# MobileNetV2 pré-entraîné
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # on ne l'entraîne pas dans un premier temps

# Ajout des couches de classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Entraînement
model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# Évaluation
loss, acc = model.evaluate(test_generator)
print(f"✅ Test accuracy: {acc:.2f}")

# Sauvegarde
os.makedirs("model", exist_ok=True)
model.save("model/dog_breed_classifier_mobileNet.h5")
