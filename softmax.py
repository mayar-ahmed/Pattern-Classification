from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization
from keras import regularizers
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard,ModelCheckpoint
from time import time
import os
import numpy as np
from features import *
expno=8
directory="linear/experiments/exp{}/checkpoints/".format(expno)
if not os.path.exists(directory):
    os.makedirs(directory)

x_train=np.load("data/x_train.npy")
y_train=np.load("data/y_train.npy")
x_val=np.load("data/x_val.npy")
y_val=np.load("data/y_val.npy")

x_sm=np.load("data_1000/x_train1.npy")
y_sm=np.load("data_1000/y_train1.npy")
x_v=np.load("data_1000/x_val1.npy")
y_v=np.load("data_1000/y_val1.npy")

#extract hog deatures for xtrain and validation
xt_feats=np.load("data/xt_feats.npy")
xv_feats=np.load("data/xv_feats.npy")

# mean_feat = np.mean(xt_feats, axis=0, keepdims=True)
# xt_feats -= mean_feat
# xv_feats -= mean_feat
#
# # Preprocessing: Divide by standard deviation. This ensures that each feature
# # has roughly the same scale.
# std_feat = np.std(xt_feats, axis=0, keepdims=True)
# xt_feats /= std_feat
# xv_feats /= std_feat

# Preprocessing: Add a bias dimension
X_train = np.hstack([xt_feats, np.ones((xt_feats.shape[0], 1))])
X_val = np.hstack([xv_feats, np.ones((xv_feats.shape[0], 1))])

print(X_train.shape)
print(X_val.shape)
model = Sequential()

#kernal initialier is uniform
#model.add(Flatten(input_shape=(256,256,1)))
model.add(Dense(3,kernel_regularizer=regularizers.l2(1),bias_regularizer=regularizers.l2(1), input_shape=(9217,)))
model.add(Activation('softmax'))

#checkpoint

filepath="linear/experiments/exp{}/checkpoints/weights-best-{{val_acc:.2f}}.hdf5".format(expno)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
tensorboard=TensorBoard(log_dir="linear/experiments/exp{}/summaries".format(expno))

sgd=SGD(lr=5e-5)

model.compile(optimizer=sgd,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,validation_data=(X_val,y_val),shuffle=True,batch_size=32,callbacks=[tensorboard,checkpoint],epochs=500)