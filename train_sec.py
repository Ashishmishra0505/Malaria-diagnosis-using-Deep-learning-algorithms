"""
Adapted from keras example cifar10_cnn.py
Train ResNet-18 on the CIFAR10 small images dataset.
GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10.py
"""
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
import numpy as np
from scipy.misc import imread
from keras.models import load_model
import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import random
import cv2
import time
import h5py

import numpy as np
import train_resnet



EPOCHS = 4
INIT_LR = 1e-3
BS = 32


def preprocessing_and_training(EPOCHS,INIT_LR,BS):
 
  # initialize the data and labels
  print("[INFO] loading images...")
  data = []
  labels = []
   
  # grab the image paths and randomly shuffle them

  path="testdata_resnet/"
  df=pd.DataFrame.from_csv("malaria_dataset_resnet.csv")


  # test=test.drop('label',1)

  temp = []
  for img_name in df.filename:
    image_path = path+img_name
    img = cv2.imread(image_path)
    img = img.reshape(1, 64 * 64 * 3)
    img = img.astype('float32')
    temp.append(img)


  train_x = np.stack(temp)

  train_x /= 255.0

  train_x = train_x.reshape(len(df), 64,64,3).astype('float32')
  train_y = to_categorical(df.label.values,num_classes=2)

  print("no_prob")
  train_x=np.array(train_x)
  train_y=np.array(train_y)

  x_train,x_test,y_train,y_test=train_test_split(train_x,train_y,test_size=0.2,random_state=4)
  # time.sleep(10)

  aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")


  aug.fit(train_x)

  print("[INFO] compiling model...")

  model = train_resnet.ResnetBuilder.build_resnet_18((3, 64, 64), 2)
  opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
  model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
  # train the network
  print("[INFO] training network...")
  H = model.fit_generator(aug.flow(x_train, y_train, batch_size=BS),validation_data=(x_test, y_test),steps_per_epoch=len(x_train) // BS,epochs=EPOCHS, verbose=1)
   
  # save the model to disk
  print("[INFO] serializing network...")
  # model.save('face_reg.h5')
  plot_graph(H,EPOCHS,INIT_LR,BS)


def plot_graph(H,EPOCHS,INIT_LR,BS):

  plt.style.use("ggplot")
  plt.figure()
  N = EPOCHS
  print(H.history["acc"])
  plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
  plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
  plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
  plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
  plt.title("Training Loss and Accuracy on our system")
  plt.xlabel("Epoch #")
  plt.ylabel("Loss/Accuracy")
  plt.legend(loc="lower left")
  # plt.savefig(args["plot"])
  plt.show()

preprocessing_and_training(EPOCHS,INIT_LR,BS)

