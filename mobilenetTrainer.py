import tensorflow as tf
import pandas as pd
import numpy as np
import os
import keras
import time
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session
from tensorflow.keras.callbacks import TensorBoard


config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

sess = tf.Session(config=config)
set_session(sess)

imageAugmentation = False

base_model=MobileNet(input_shape=[224,224,3], weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(9,activation='softmax')(x) #final layer with softmax activation

model=Model(inputs=base_model.input,outputs=preds)
#specify the inputs
#specify the outputs
#now a model has been created based on our architecture

train_from_layer = 86

for i,layer in enumerate(model.layers):
  print(i,layer.name)

#for layer in model.layers:
 #   layer.trainable=False

#for layer in model.layers[:train_from_layer]:
 #   layer.trainable=False
#for layer in model.layers[train_from_layer:]:
 #   layer.trainable=True

train_data_dir = '/home/csabee005/Documents/SmartFridge'

batch_size = 64
epochs = 1
validation_split = 0.2
step_size_train = 150

NAME = "Smart_Fridge_BaseSet_imageAugmentation:{}_batchSize:{}_epochs:{}_valSplit:{}_stepSize:{}_trainedFromLayer:{}_time:{}".format(imageAugmentation, batch_size, epochs, validation_split, step_size_train, train_from_layer ,int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

train_datagen=ImageDataGenerator(rescale=1./255, validation_split= validation_split)

if imageAugmentation == True:
    train_datagen.height_shift_range = 0.2
    train_datagen.width_shift_range = 0.2
    train_datagen.zoom_range = zoom_range=[0.1,0.3]
    train_datagen.shear_range = 30



train_generator=train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(224,224),
            color_mode='rgb',
            batch_size=batch_size,
            class_mode='categorical',
            subset= 'training',
            shuffle=True)



validation_generator = train_datagen.flow_from_directory(
            train_data_dir, # same directory as training data
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=True)

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   validation_data=validation_generator,
                   validation_steps = step_size_train,
                   callbacks=[tensorboard],
                   epochs=epochs)

model.save('mobileNet.model')