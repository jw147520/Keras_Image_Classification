# Keras_Image_Classification
Retraining pre-trained Inception v3 for image classification on different domain

Requirements
1) tensorflow
2) keras
3) cv2
4) Augmentor - https://github.com/mdbloice/Augmentor
5) numpy

Usages
1) data_augmentation.py
  python data_augmentation.py [path]
2) load_data.py
  batch generator - called by train.py, train_layer.py, test.py
3) train.py, train_layer.py
  python train.py or python train_layer.py
4) test.py
  python test.py
  
  
