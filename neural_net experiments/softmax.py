from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization
from keras import regularizers
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard,ModelCheckpoint
from time import time
import os
import numpy as np
from HOG import *
expno=3
directory="linear/experiments/exp{}/checkpoints/".format(expno)
if not os.path.exists(directory):
    os.makedirs(directory)

x_train=np.load("../data/xt_feats.npy")
y_train=np.load("../data/yt_aug.npy")
x_val=np.load("../data/xv_feats.npy")
y_val=np.load("../data/y_val.npy")

mean_feat= np.mean(x_train, axis=0)
std=np.std(x_train, axis=0)

# x_train-=mean_feat
# x_val-=mean_feat
# x_train/=std
# x_val/=std

reg=5e-4
model = Sequential()

#kernal initialier is uniform
#model.add(Flatten(input_shape=(256,256,1)))
model.add(Dense(3,kernel_regularizer=regularizers.l2(reg), input_shape=(1764,)))
model.add(Activation('softmax'))

#checkpoint

filepath="linear/experiments/exp{}/checkpoints/weights-best-{{val_acc:.2f}}.hdf5".format(expno)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
tensorboard=TensorBoard(log_dir="linear/experiments/exp{}/summaries".format(expno))

sgd=SGD(lr=8e-3)

model.compile(optimizer=sgd,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,validation_data=(x_val,y_val),shuffle=True,batch_size=16,callbacks=[tensorboard,checkpoint],epochs=300)
model.save('linear/model.h5')