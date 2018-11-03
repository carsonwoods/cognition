import numpy as np

from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

test_model = load_model('./models/cognition_model.h5')
img = load_img('./data/CatBenchmark.jpeg',False,target_size=(150,150))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
preds = test_model.predict_classes(x)
probs = test_model.predict_proba(x)
print(preds, probs)
