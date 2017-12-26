# -*- coding: utf-8 -*-
# Data Augmentation : Augmentor library 사용.
# Augmentor : Image augmentation library in python for machine learning.
# https://github.com/mdbloice/Augmentor
import Augmentor
import os
import sys


# usage: python data_augmentation.py [path]

path = sys.argv[1]  # 실행 인자로 전달받은 원본 이미지 경로 (./data)

for dir_name in os.listdir(path):  # 원본 이미지 경로에 있는 모든 하위 디렉토리(즉 모든 label)
	p = Augmentor.Pipeline(path + '/' + dir_name)  # image file 들이 저장된 directory 를 넘겨준다.
	num_images = len(p.augmentor_images)

	# 90도 회전한 이미지
	# p.rotate90(probability=1)
	# p.sample(num_images)
	# 180도 회전한 이미지
	# p.rotate180(probability=1)
	# p.sample(num_images)
	# 270도 회전한 이미지
	# p.rotate270(probability=1)
	# p.sample(num_images)
	# 좌우 뒤집은 이미지
	# p.flip_left_right(probability=1)
	# p.sample(num_images)
	# 상하 뒤집은 이미지
	# p.flip_top_bottom(probability=1)
	# p.sample(num_images)
	# random distortion 을 적용한 이미지
	p.random_distortion(probability=1, grid_width=10, grid_height=10, magnitude=1)
	p.sample(num_images * 3)  # random distortion을 한 샘플 3개를 생성

# 이미지 한 장당 8 장의 augmented images 가 생성되어 원래 data set 의 9배로 부풀림.
