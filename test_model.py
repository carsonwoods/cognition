import numpy as np
import os.path

# Sets random seed to constant for reproducibility
np.random.seed(123)

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.utils import np_utils
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model


classifier = load_model('./first_try.h5')
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])


test_image1 = image.load_img("./data/BenchmarkImage.jpeg", target_size = (128, 128))
test_image1 = image.img_to_array(test_image1)
test_image1 = np.expand_dims(test_image1, axis = 0)

test_image2 = image.load_img("./data/BenchmarkImage1.jpeg", target_size = (128, 128))
test_image2 = image.img_to_array(test_image2)
test_image2 = np.expand_dims(test_image2, axis = 0)

#predict the result
result = classifier.predict(test_image1)
result2 = classifier.predict(test_image2)
print("Image 1", result[0])
print("Image 2", result[0])
