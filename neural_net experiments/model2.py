from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization
from keras.optimizers import Adam,RMSprop,SGD
from keras.callbacks import TensorBoard,ModelCheckpoint
from time import time
import os
import numpy as np
from keras.utils import plot_model
expno=3
directory="model2/experiments/exp{}/checkpoints/".format(expno)
if not os.path.exists(directory):
    os.makedirs(directory)

x_train=np.load("../data/xt_aug.npy")
y_train=np.load("../data/yt_aug.npy")

x_val=np.load("../data/x_val.npy")
x_val=x_val.reshape(x_val.shape + (1,))
y_val=np.load("../data/y_val.npy")

#suffle data
N= x_train.shape[0]
ind_list = [i for i in range(N)]
np.random.shuffle(ind_list)
x_train = x_train[ind_list, :,:,:]
y_train = y_train[ind_list,]


model = Sequential()

model.add(Conv2D(32,(5,5), strides=1, padding='same',input_shape=(256, 256,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3) ,strides=1, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3) ,strides=1, padding='same' ))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3) ,strides=1, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3) ,strides=1, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.1))


model.add(Dense(3))
model.add(Activation('softmax'))

#checkpoint
print(model.summary())

filepath="model2/experiments/exp{}/checkpoints/weights-best-{{val_acc:.2f}}.hdf5".format(expno)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
tensorboard=TensorBoard(log_dir="model2/experiments/exp{}/summaries".format(expno))

rms=RMSprop(lr=1e-3)

model.compile(optimizer=rms,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,validation_data=(x_val,y_val),batch_size=32,shuffle=True,callbacks=[tensorboard,checkpoint],epochs=20)
model.save("model2/model.h5")

plot_model(model, to_file='model2im.png',show_layer_names=False , show_shapes=False)