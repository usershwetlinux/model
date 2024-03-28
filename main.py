import os
import numpy as np
import shutil
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image as tf_image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Define your dataset paths

# Load pre-trained VGG16 model without the top classification layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classification layers for fine-tuning
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # You may adjust the number of units based on your dataset
num_classes=7
predictions = Dense(num_classes, activation='softmax')(x)  # Define num_classes based on your dataset

# Combine base model and new classification layers
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the pre-trained layers to prevent them from being updated during training
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data preparation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory("D:\\ml project\\dataset\\train",

    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    "D:\\ml project\\dataset\\validate",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

# Fine-tune the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=50,  # Adjust the number of epochs as needed
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32)

# Save the fine-tuned model
model.save("fine_tuned_aesthetics_model.h5")
