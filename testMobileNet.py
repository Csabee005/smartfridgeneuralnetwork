import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np

def prepare(filepath):
    IMG_SIZE = 224
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array/255, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

CATEGORIES = ["Apple","Banana","Guava","Kiwi","Orange","Peach","Pear","Plum","Tomatoes"]

model = tf.keras.models.load_model("mobileNet.model")

predictions = model.predict([prepare('Kiwi002.png')])
print(predictions[0])
print(CATEGORIES[int(predictions.argmax(axis=-1))])

predictions = model.predict([prepare('Kiwi0018.png')])
print(predictions[0])
print(CATEGORIES[int(predictions.argmax(axis=-1))])

predictions = model.predict([prepare('Kiwi0044.png')])
print(predictions[0])
print(CATEGORIES[int(predictions.argmax(axis=-1))])

predictions = model.predict([prepare('Kiwi00109.png')])
print(predictions[0])
print(CATEGORIES[int(predictions.argmax(axis=-1))])

predictions = model.predict([prepare('Kiwi00180.png')])
print(predictions[0])
print(CATEGORIES[int(predictions.argmax(axis=-1))])

predictions = model.predict([prepare('Kiwi00219.png')])
print(predictions[0])
print(CATEGORIES[int(predictions.argmax(axis=-1))])

predictions = model.predict([prepare('Banana08.png')])
print(predictions[0])
print(CATEGORIES[int(predictions.argmax(axis=-1))])

predictions = model.predict([prepare('Banana022.png')])
print(predictions[0])
print(CATEGORIES[int(predictions.argmax(axis=-1))])

predictions = model.predict([prepare('Banana0105.png')])
print(predictions[0])
print(CATEGORIES[int(predictions.argmax(axis=-1))])

predictions = model.predict([prepare('Banana0185.png')])
print(predictions[0])
print(CATEGORIES[int(predictions.argmax(axis=-1))])

predictions = model.predict([prepare('Banana0256.png')])
print(predictions[0])
print(CATEGORIES[int(predictions.argmax(axis=-1))])

#print(predictions)
#print(CATEGORIES[np.argmax(predictions[0])])

predictions = model.predict([prepare('Tamotoes003.png')])
print(CATEGORIES[int(predictions.argmax(axis=-1))])

#print(predictions)
#print(CATEGORIES[np.argmax(predictions[0])])

predictions = model.predict([prepare('Tamotoes003.png')])
print(CATEGORIES[int(predictions.argmax(axis=-1))])

predictions = model.predict([prepare('Peach0018.png')])
print(CATEGORIES[int(predictions.argmax(axis=-1))])

#print(predictions)
#print(CATEGORIES[np.argmax(predictions[0])])