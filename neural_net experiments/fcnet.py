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
from HOG import *

from keras.utils import plot_model
reg=5e-3
expno=4
directory="fcnet/experiments/exp{}/checkpoints/".format(expno)
if not os.path.exists(directory):
    os.makedirs(directory)

x_train=np.load("../data/xt_feats.npy")
y_train=np.load("../data/yt_aug.npy")
x_val=np.load("../data/xv_feats.npy")
y_val=np.load("../data/y_val.npy")

# mean_feat= np.mean(x_train, axis=0)
# std=np.std(x_train, axis=0)
#
# x_train-=mean_feat
# x_val-=mean_feat
# x_train/=std
# x_val/=std
#


model = Sequential()


model.add(Dense(512,input_shape=(1764,) ,kernel_regularizer=regularizers.l2(reg)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(256,kernel_regularizer=regularizers.l2(reg) ))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(64,kernel_regularizer=regularizers.l2(reg)))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(32,kernel_regularizer=regularizers.l2(reg) ))
model.add(Activation("relu"))
model.add(Dropout(0.2))


model.add(Dense(3,kernel_regularizer=regularizers.l2(reg) ))
model.add(Activation("softmax"))

print(model.summary())
#checkpoint

filepath="fcnet/experiments/exp{}/checkpoints/weights-best-{{val_acc:.2f}}.hdf5".format(expno)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
tensorboard=TensorBoard(log_dir="fcnet/experiments/exp{}/summaries".format(expno))

adam=Adam(1e-3)

model.compile(optimizer=adam,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,validation_data=(x_val,y_val),shuffle=True,batch_size=32,callbacks=[tensorboard,checkpoint],epochs=20)
model.save("fcnet/model2.h5")

plot_model(model, to_file='fcmodelim.png',show_layer_names=False , show_shapes=False)