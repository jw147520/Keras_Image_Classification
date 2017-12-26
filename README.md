# Keras_Image_Classification
## Retraining pre-trained Inception v3 for image classification on different domain
----------

Requirements
-------------

>- tensorflow
>- keras
>- cv2
>- Augmentor - https://github.com/mdbloice/Augmentor
>- numpy

####  Usages
Input image는 하위 폴더에 class 별로 정리되어 있어야 한다.
ex) ./data/dogs,  ./data/cats , ... etc.

1. data_augmentation.py
    
    python data_augmentation.py  [path]
2. load_data.py
    
    batch_generator - called by train.py, train_layer.py, test.py
3. train.py, train_layer.py
    
    python train.py(or train_layer.py)
4. test.py
    
    python test.py

----------


