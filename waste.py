import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)
val_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
        "/Users/wrath/Bayu/00 - Bangkit/WasteWizard/Bangkit2023/data/waste_datasets/DATASET/TRAIN",
        target_size = (150, 150),
        batch_size = 128,
        class_mode = 'binary')

val_generator = val_datagen.flow_from_directory(
        "/Users/wrath/Bayu/00 - Bangkit/WasteWizard/Bangkit2023/data/waste_datasets/DATASET/TRAIN",
        target_size = (150, 150),
        batch_size = 128,
        class_mode = 'binary')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss = "binary_crossentropy",
              optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=0.001),
              metrics = ["accuracy"])

model.fit(
      train_generator,
      steps_per_epoch = 176, # 22,564 images = batch_size * steps
      epochs = 10,
      verbose = 1,
      validation_data = val_generator, # 4,513 images = batch_size * steps
      validation_steps = 36)