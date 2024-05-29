import pandas as pd
import requests
from sklearn.model_selection import train_test_split
import os
import pickle
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from matplotlib.image import imread
from keras.preprocessing import image

def download_img(csv_path, download_path):
  data_path = pd.read_csv(csv_path)
  data_path = data_path.dropna()
  urls = list(data_path['Woo image back'])
  labels = data_path['Condition']
  label_type = list(labels.unique())

  if not os.path.exists(download_path):
    os.mkdir(download_path)
  
  for label in label_type:
    if not os.path.exists(os.path.join(download_path, label)):
      os.mkdir(os.path.join(download_path, label))
  
  for i in range(len(urls)):
    img_data = requests.get(urls[i]).content
    with open(os.path.join(download_path, list(labels)[i], urls[i].split('/')[-1]), 'wb') as handler:
      handler.write(img_data)
      print('download from ', urls[i])

def split_train_test(img_path, test_size, batch_size):
  train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=img_path,
    validation_split=test_size,
    subset="training",
    seed=123,
    image_size=(255, 255),
    batch_size=batch_size)
  val_ds = tf.keras.utils.image_dataset_from_directory(
    directory=img_path,
    validation_split=test_size,
    subset="validation",
    seed=123,
    image_size=(255, 255),
    batch_size=batch_size)
  return train_ds, val_ds

def train(download=False, csv_path='', download_path='', img_path='', test_size=0.1, batch_size=32, epochs=1):
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  if download:
    download_img(csv_path, download_path)
    img_path = download_path

  train_ds, val_ds = split_train_test(img_path=img_path, test_size=test_size, batch_size=batch_size)

  VGG16_MODEL=tf.keras.applications.VGG16(input_shape=[255, 255, 3],
                                          include_top=False,
                                          weights='imagenet')
  VGG16_MODEL.trainable=False
  global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
  prediction_layer = tf.keras.layers.Dense(len(train_ds.class_names),activation='softmax')
  model = tf.keras.Sequential([
    VGG16_MODEL,
    global_average_layer,
    prediction_layer
  ])

  model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
  
  print(model.summary)

  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)
  
  model.save(os.path.join(os.getcwd(), 'my_model.h5'))
	
  with open(os.path.join(os.getcwd(), 'label.txt'), 'w') as f:
    for label in list(train_ds.class_names):
      f.write(str(label)+',')

  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig(os.path.join(os.getcwd(), 'accuracy.png'))

if __name__ == "__main__":
	train(download=False, csv_path='', download_path='', img_path='', test_size=0.1, batch_size=32, epochs=1)
