from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization
from keras.optimizers import Adam,RMSprop,SGD
from keras.callbacks import TensorBoard,ModelCheckpoint
from time import time
from keras import regularizers

import os
import numpy as np

expno=12
reg=0.05
directory="model1/experiments/exp{}/checkpoints/".format(expno)
if not os.path.exists(directory):
    os.makedirs(directory)

#load augmented training data and original validation data
x_train=np.load("../data/xt_aug.npy")
y_train=np.load("../data/yt_aug.npy")

x_val=np.load("../data/x_val.npy")
x_val=x_val.reshape(x_val.shape + (1,))
print("x_val after reshaping" ,x_val.shape)
y_val=np.load("../data/y_val.npy")


model = Sequential()

#kernal initialier is uniform
model.add(Conv2D(32,(5,5), strides=2,input_shape=(256, 256,1), kernel_regularizer=regularizers.l2(reg)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3) ,strides=1,kernel_regularizer=regularizers.l2(reg)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3) ,strides=1,kernel_regularizer=regularizers.l2(reg) ))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),))

model.add(Conv2D(32,(3,3) ,strides=1,kernel_regularizer=regularizers.l2(reg)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(128,kernel_regularizer=regularizers.l2(reg)))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(64,kernel_regularizer=regularizers.l2(reg)))
model.add(Activation('relu'))
model.add(Dropout(0.1))


model.add(Dense(32,kernel_regularizer=regularizers.l2(reg)))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(3,kernel_regularizer=regularizers.l2(reg)))
model.add(Activation('softmax'))

#checkpoint
print(model.summary())

filepath="model1/experiments/exp{}/checkpoints/weights-best-{{val_acc:.2f}}.hdf5".format(expno)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
tensorboard=TensorBoard(log_dir="model1/experiments/exp{}/summaries".format(expno))

adam=RMSprop(lr=1e-3)

model.compile(optimizer=adam,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,validation_data=(x_val,y_val), shuffle=True, batch_size=32,callbacks=[tensorboard,checkpoint],epochs=20)