# -*- coding: utf-8 -*- #
import sys
from keras.models import load_model
import os
import random
import cv2
import numpy as np


data_path = ''  # input data path

model = load_model('model.h5')
print("Successfully loaded model")
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

if input("Continue to predict?") == 'y':

	print("Loading input data...")
	class_list = list()  # class list
	data_pair = list()  # list of [file name - class]

	for dir in os.listdir(data_path):
		print(dir)
		# data path 의 하위 디렉토리(class 별로 분류된)를 탐색
		class_list.append(dir)  # append dir(class) to class list

		path = os.path.join(data_path, dir)
		for file in os.listdir(path):
			# class 별 디렉토리의 하위 파일들을 탐색
			pair = [os.path.join(path, file), dir]  # [file path, label]
			data_pair.append(pair)

	random.shuffle(data_pair)
	print("Finished loading input data")
	print("Total test data size is " + str(len(data_pair)))

	index = 0
	num_hit = 0
	num_miss = 0
	
	while index < len(data_pair):
		try:
			pair = data_pair[index]
			img = cv2.imread(pair[0], cv2.IMREAD_COLOR)
			img = cv2.resize(img, (299, 299))
			label = pair[1]

			input_img = np.reshape(img, (1,) + img.shape)
			prediction = model.predict(input_img)
			predicted_label = class_list[np.argmax(prediction)]
			real_label = label

			result = "Hit!!"
			if predicted_label != real_label:
				result = "Miss!!"
				num_miss += 1
			else:
				num_hit += 1

			print("Test image " + str(index + 1))
			for i in range(len(class_list)):
				print("Prob. " + str(class_list[i]) + ' ' + str(prediction[0][i]))
			print("prediction: " + str(predicted_label) + "    real: " + str(real_label) + "  -->  " + result)
			print("")

			index += 1

		except Exception as e:
			print("Excepted with " + str(e))
			break

	print("----------------------------------------")
	print("Total : " + str(len(data_pair)))
	print("Hit : " + str(num_hit))
	print("Miss : " + str(num_miss))
	print("Accuracy : " + str(num_hit / len(data_pair)))

else:
	print("Stop testing...")



