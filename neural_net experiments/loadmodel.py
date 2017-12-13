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

#utility functions for images

#load image into numpy array
def img2array(path):
    #reads single image into numpy array
    img = misc.imread(path)
    s=np.array(img)
    s=s.reshape(s.shape + (1,))
    return s

#laod folder of images (uses img2array)
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

#takes numy array of data abd outputs image with its clss

def save_output(x,predicted,path):
    #saves predicted output to
    x=np.reshape(x, (-1,256,256))
    for i in range(x.shape[0]):

        im = Image.fromarray(x[i])
        im.save(path+str(i)+"class"+str(predicted[i])+ ".png")

def convnet_model2():

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
    return model

def fc_model():
    pass

def convnet_model():
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


#load data from test file and appropriate model and return accuracy, save outputs in a file
def test_model(m,image_path, weight_path, outpath):

    #load validation data manually just for checking
    #saved images in data folder are 256,256 only
    x_val=np.load("../data/x_val.npy")
    y_val=np.load("../data/y_val.npy")
    x_val=x_val.reshape(x_val.shape + (1,))

    #returns  (N,256,256,1) images from folder
    x= load_folder(image_path)

    model=None
    if m=='conv1':
        model=convnet_model()
        rms=RMSprop(lr=1e-3)
        model.compile(optimizer=rms,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    elif m=='conv2':
        model=convnet_model2()
        rms=RMSprop(lr=1e-3)
        model.compile(optimizer=rms,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    elif model=='fc':
        model=fc_model()
    else:
        print("invalid model")
        return

    model.load_weights(weight_path)
    print("Created model and loaded weights from file")
    #call function setup model here to compile it according to original mode;



    scores = model.evaluate(x_val,y_val)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    predictions = model.predict(x_val,verbose=True)
    predicted_class=np.argmax(predictions,axis=1)
    save_output(x_val,predicted_class,outpath )
    diff= y_val-predicted_class
    print(diff)
    return predicted_class


def main():
    #path to test folder
    path="../validation_data"
    weight_path="/media/mayar/not_fun/year4/pattern/image classifier/neural_net experiments/model1/experiments/exp12/checkpoints/weights-best-0.97.hdf5"
    p= test_model('conv1',path,weight_path,'results/')



main()
