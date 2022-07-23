import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import os

from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

!kaggle datasets download -d tongpython/cat-and-dog

import tensorflow as tf
from urllib.request import urlretrieve
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

zip_ref = zipfile.ZipFile(r'/content/cat-and-dog.zip', 'r') #Opens the zip file in read mode
zip_ref.extractall('/home') #Extracts the files into the /tmp folder
zip_ref.close()

len(os.listdir('/home/training_set/training_set/dogs/'))

training_dogs = ('/home/training_set/training_set/dogs')

training_dir = '/home/training_set/training_set/'
validation_dir = '/home/test_set/test_set'

Batch_size = 100
img_shape = 300

train_datagen = ImageDataGenerator(rescale=1./255,
  rotation_range=40,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  fill_mode='nearest'
 )

train_data_gen = train_datagen.flow_from_directory(directory=training_dir,
                                                           target_size=(img_shape, img_shape),
                                                           batch_size=Batch_size,
                                                           shuffle=True,
                                                           class_mode='binary')

validation_image_generator = ImageDataGenerator(rescale=1./255)

validation_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir, 
                                                                     target_size=(img_shape, img_shape),
                                                                     batch_size=Batch_size, 
                                                                     class_mode='binary',
                                                                     shuffle=False)

from os import listdir
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

model = tf.keras.Sequential([
                             tf.keras.layers.Conv2D(16, (3,3), input_shape=(img_shape, img_shape, 3), activation=tf.nn.relu),
                             tf.keras.layers.MaxPooling2D(2,2),
                             tf.keras.layers.Conv2D(16, (3,3), activation=tf.nn.relu),
                             tf.keras.layers.MaxPooling2D(2,2),
                             tf.keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu),
                             tf.keras.layers.MaxPooling2D(2,2),
                             tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu),
                             tf.keras.layers.MaxPooling2D(2,2),
                             tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu),
                             tf.keras.layers.MaxPooling2D(2,2),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(512, activation=tf.nn.relu),
                             tf.keras.layers.Dense(2)
])

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

Epochs = 15
history = model.fit(train_data_gen,
                    epochs=Epochs,
                    validation_data=validation_data_gen
                  )

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(Epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('./foo.png')
plt.show()

predictions = model.predict(validation_data_gen)

predictions[0]

predictions= np.argmax(predictions[1])

predictions

from tensorflow.keras.models import save_model
save_model(model, 'model1.h5')

import numpy as np
from google.colab import files 
from keras.preprocessing import image

uploaded=files.upload()

for fn in uploaded.keys():

# predicting images

  path='/content/' + fn

  img=image.load_img(path, target_size=(300, 300))

  x = image.img_to_array(img)

  x= np.expand_dims(x, axis=0)

  image_tensor = np.vstack([x])

  classes=model.predict(image_tensor)
  print(classes * 10)
  if classes[0][0]>20:
    print(fn + 'is a cat')
  else:
    classes[0][1]>20
    print(fn + 'is a dog')

