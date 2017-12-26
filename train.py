# -*- coding: utf-8 -*-
import keras
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.utils import plot_model
from keras.optimizers import SGD
from keras import backend as K
import numpy as np
from load_data import *


nb_classes = 3  # number of classes
data_path = ''  # input data path

# input image size
img_width = 299
img_height = 299
img_channel = 3
img_shape = (img_width, img_height, img_channel)
batch_size = 32  # batch size

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=img_shape)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(nb_classes, activation='softmax')(x)

# this is the model to train
model = Model(inputs=base_model.input, outputs=predictions)
# model = load_model( ... )  # if want to train using pre-trained model

# train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV2 layers
for layer in base_model.layers:
	layer.trainable = False

# compile the model (should be done "after" setting layers to non-trainable)
model.compile(optimizer=SGD(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
with open('ModelSummary.txt', 'w') as fh:
	model.summary(print_fn=lambda x: fh.write(x + '\n'))

# log files for training/validation loss, accuracy
loss_training_file = open('log/loss_training.txt', 'w')
loss_val_file = open('log/loss_val.txt', 'w')
acc_training_file = open('log/acc_training.txt', 'w')
acc_val_file = open('log/acc_val.txt', 'w')


print("Start Training...")
batch_count = 0
try:
	for i in range(0, 30):
		print('--------------- On epoch: ' + str(i) + ' ---------------')
		for x_data, y_label, classlist in load_batch(data_path, batch_size):
			x_data = np.array(x_data)
			print(x_data.shape)

			history = model.fit(x_data, y_label, verbose=1, epochs=1, validation_split=.2, batch_size=batch_size)
			batch_count += 1

			# log for training/validation loss, accuracy
			loss_training_file.write(str(history.history['loss'][0]) + '\n')
			loss_val_file.write(str(history.history['val_loss'][0]) + '\n')
			acc_training_file.write(str(history.history['acc'][0]) + '\n')
			acc_val_file.write(str(history.history['val_acc'][0]) + '\n')


		print("Saving checkpoint on epoch " + str(i))
		model.save('model_chkp_' + str(i) + '.h5')
		print("Checkpoint saved. Continuing...")

		model.save('model.h5')
except Exception as e:
	print("Excepted with " + str(e))
	print("Saving model...")
	model.save('excepted_model.h5')
	print("Model saved.")

# save final model
model.save('model.h5')
print("Model saved. Finished training.")



