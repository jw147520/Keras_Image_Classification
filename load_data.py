# -*- coding: utf-8 -*-
import os
import random
import cv2
from keras.utils import np_utils

# data_path = '../data/2classes/'  # data path
# data path 아래에 class 별로 구분된 이미지 폴더가 존재해야 한다.

# Batch Generator
# Load images from given data path and yields batch of given size
def load_batch(data_path, batch_size):
	"""
	:param data_path: data directory path which input data is stored
	:param batch_size: batch size
	:return: generated batches of input data
	"""
	print("Loading dataset file...")
	data_path = data_path

	class_list = list()  # class list
	# file_dic = dict()  # dictionary of "class name": [file list]
	data_pair = list()  # list of [file name - class]

	for dir in os.listdir(data_path):
		# data path 의 하위 디렉토리(class 별로 정리된)를 탐색
		class_list.append(dir)  # append dir(class) to class list
		# file_dic[dir] = list()  # 새로 들어온 key(class name)의 value 로 list 를 만든다.

		path = os.path.join(data_path, dir)
		for file in os.listdir(path):
			# class 별 디렉토리 하위 파일들을 탐색
			pair = [os.path.join(path, file), dir]  # [file path, label]
			data_pair.append(pair)
			# file_dic[dir].append(os.path.join(path, file))

	# shuffle data
	random.shuffle(data_pair)

	print("Finished loading dataset file.")
	print("Total Data Size is " + str(len(data_pair)))

	batch_count = 1
	index = 0
	while index < len(data_pair):
		try:
			x_data = []
			y_label = []
			count = 0

			print("---------- On Batch: " + str(batch_count) + " ----------")
			while count < batch_size and index < len(data_pair):
				pair = data_pair[index]
				img = cv2.imread(pair[0], cv2.IMREAD_COLOR)
				img = cv2.resize(img, (299, 299))

				# one-hot encoding
				encoded = []
				for i in range(len(class_list)):
					if i == class_list.index(pair[1]):
						encoded.append(1.)
					else:
						encoded.append(0.)

				# Train / Test split은 model.fit 에서 validation_split 을 통해서 한다.
				x_data.append(img)
				
				y_label.append(encoded)
				count += 1
				index += 1

			print("Batch loaded.")
			batch_count += 1
			yield x_data, y_label, class_list

		except Exception as e:
			print('Excepted while Loading Data with ' + str(e))
			break
