from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization
from keras.optimizers import Adam,RMSprop,SGD
from keras.callbacks import TensorBoard,ModelCheckpoint
from time import time
import os
import numpy as np
from scipy import misc
from PIL import Image
import glob

def img2array(path):
    #reads single image into numpy array
    img = misc.imread(path)
    s=np.array(img)
    s=s.reshape(s.shape + (1,))
    return s


def load_folder(folder_path):
    #takes paths and returns its images as numpy array
    x=[]
    for path in glob.glob(folder_path+"/*.png"):
        s=img2array(path)
        if s.shape==(256,256,3):
            continue
        x.append(img2array(path))


    m=np.stack(x)
    return m

def create_model():
    model = Sequential()
    #kernal initialier is uniform
    model.add(Conv2D(32,(5,5), strides=2,input_shape=(256, 256,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32,(3,3) ,strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32,(3,3) ,strides=1 ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32,(3,3) ,strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))


    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(3))
    model.add(Activation('softmax'))

    return model

#checkpoint

def test_model(x, weight_path):

    model=create_model()
    model.load_weights(weight_path)
    rms=RMSprop(lr=1e-3)
    model.compile(optimizer=rms,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    print("Created model and loaded weights from file")
    predictions = model.predict(x,verbose=True)
    return predictions


def main():
    #path to test folder
    path="../validation_data"
    x_test = load_folder(path)
    p= test_model(x_test,"/media/mayar/not_fun/year4/pattern/image classifier/neural_net experiments/model1/experiments/exp11/checkpoints/weights-best-0.94.hdf5")
    print(p)
    print(p.shape)
    correct=np.argmax(p,axis=1)
    print(correct)

main()
