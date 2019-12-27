from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation, Input
import pandas as pd
import os
import numpy as np
from scipy.misc import imread
from keras.models import load_model
import tensorflow as tf
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
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
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

class AlexNetmodel:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model
		model = Sequential()

		# 1st Convolutional Layer
		model.add(Conv2D(filters=96, input_shape=(64,64,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
		model.add(Activation('relu'))
		# Max Pooling
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

		# 2nd Convolutional Layer
		model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
		model.add(Activation('relu'))
		# Max Pooling
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

		# 3rd Convolutional Layer
		model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
		model.add(Activation('relu'))

		# 4th Convolutional Layer
		model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
		model.add(Activation('relu'))

		# 5th Convolutional Layer
		model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
		model.add(Activation('relu'))
		# Max Pooling
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

		# Passing it to a Fully Connected layer
		model.add(Flatten())
		# 1st Fully Connected Layer
		model.add(Dense(4096, input_shape=(64*64*3,)))
		model.add(Activation('relu'))
		# Add Dropout to prevent overfitting
		model.add(Dropout(0.4))

		# 2nd Fully Connected Layer
		model.add(Dense(4096))
		model.add(Activation('relu'))
		# Add Dropout
		model.add(Dropout(0.4))

		# 3rd Fully Connected Layer
		model.add(Dense(1000))
		model.add(Activation('relu'))
		# Add Dropout
		model.add(Dropout(0.4))

		# Output Layer
		model.add(Dense(classes))
		model.add(Activation('sigmoid'))

		# return the constructed network architecture
		return model

def alexnet_model(img_shape=(64, 64, 3), n_classes=2, l2_reg=0.,
	weights=None):

	# Initialize model
	alexnet = Sequential()

	# Layer 1
	alexnet.add(Conv2D(96, (11, 11), input_shape=img_shape,
		padding='same', kernel_regularizer=l2(l2_reg)))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 2
	alexnet.add(Conv2D(256, (5, 5), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 3
	alexnet.add(ZeroPadding2D((1, 1)))
	alexnet.add(Conv2D(512, (3, 3), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 4
	alexnet.add(ZeroPadding2D((1, 1)))
	alexnet.add(Conv2D(1024, (3, 3), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))

	# Layer 5
	alexnet.add(ZeroPadding2D((1, 1)))
	alexnet.add(Conv2D(1024, (3, 3), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 6
	alexnet.add(Flatten())
	alexnet.add(Dense(3072))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(Dropout(0.5))

	# Layer 7
	alexnet.add(Dense(4096))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(Dropout(0.5))

	# Layer 8
	alexnet.add(Dense(n_classes))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('sigmoid'))

	if weights is not None:
		alexnet.load_weights(weights)

	return alexnet


EPOCHS = 3
INIT_LR = 1e-3
BS = 32


def preprocessing_and_training(EPOCHS,INIT_LR,BS):
 
	# initialize the data and labels
	print("[INFO] loading images...")
	data = []
	labels = []
	 
	# grab the image paths and randomly shuffle them

	path="testdata_alexnet/"
	df=pd.DataFrame.from_csv("malaria_dataset_alexnet.csv")



	# test=test.drop('label',1)

	temp = []
	for img_name in df.filename:
		image_path = path+img_name
		img = cv2.imread(image_path)
		print(img_name)
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
	model = alexnet_model()
	opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	model.compile(loss="binary_crossentropy", optimizer=opt,
		metrics=["accuracy"])
	 
	# train the network
	print("[INFO] training network...")
	H = model.fit_generator(aug.flow(x_train, y_train, batch_size=BS),validation_data=(x_test, y_test),steps_per_epoch=len(x_train) // BS,epochs=EPOCHS, verbose=1)
	 
	# save the model to disk
	print("[INFO] serializing network...")
	model.save('alexnet.h5')
	plot_graph(H,EPOCHS,INIT_LR,BS)


def plot_graph(H,EPOCHS,INIT_LR,BS):

	plt.style.use("ggplot")
	plt.figure()
	N = EPOCHS
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy on our system")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	# plt.savefig(args["plot"])
	plt.show()

preprocessing_and_training(EPOCHS,INIT_LR,BS)