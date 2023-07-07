import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense,Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping


train_gen = ImageDataGenerator(rescale=(1/255))
validate_gen = ImageDataGenerator(rescale=(1/255))
imgSize = 224
train = train_gen.flow_from_directory('data/train',
                                      target_size=(imgSize, imgSize),
                                      class_mode='categorical',
                                      batch_size=8)
validate = validate_gen.flow_from_directory('data/validate',
                                      target_size=(imgSize, imgSize),
                                      class_mode='categorical',
                                      batch_size=8)
cnn = Sequential()
cnn.add(Convolution2D(12,(3,3),activation='relu',input_shape=(imgSize, imgSize,3)))
cnn.add(MaxPooling2D(pool_size=(2,2)))

cnn.add(Convolution2D(24,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))

cnn.add(Convolution2D(36,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))

cnn.add(Flatten())
cnn.add(Dense(256,activation='relu'))
cnn.add(Dense(128,activation='relu'))
cnn.add(Dense(64,activation='relu'))
cnn.add(Dense(26,activation='softmax'))

cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
early_stop = EarlyStopping(monitor='accuracy',patience=8)

cnn.fit(train,batch_size=8,epochs=25,callbacks=early_stop,validation_data = validate)

model_json = cnn.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
cnn.save('model.h5')