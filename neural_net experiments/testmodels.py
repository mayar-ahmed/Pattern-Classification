from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization
from keras.optimizers import Adam,RMSprop,SGD
from keras.callbacks import TensorBoard,ModelCheckpoint
from keras.models import load_model
from time import time
from features import *
import os
import numpy as np
from scipy import misc
from PIL import Image
import glob




def img2array(path):
    """

    :param path: full path of image
    :return: numpy matrix of the image (256,256)
    """
    img = misc.imread(path, mode='L')
    s=np.array(img)
    return s


def load_folder(folder_path):
    """
    loads folder of images
    :param folder_path: path which contains the images to load
    :return: numpy array of images (N,256,256,1)
    """
    x=[]
    for path in glob.glob(folder_path+"/*.bmp"):

        s=img2array(path)
        if s.shape==(256,256,3):
            print('256,256,3')
        s=s.reshape((256,256,1))
        x.append(s)

    m=np.stack(x)
    return m


def save_output(x,predicted,path):
    """

    :param x: images that were classified
    :param predicted: predicted labels
    :param path: path to output folder
    :return: none
    """
    x=np.reshape(x, (-1,256,256))
    for i in range(x.shape[0]):
        im = Image.fromarray(x[i])
        im.save(path+str(i)+"class"+str(predicted[i])+ ".png")



def model_load(model_path, weights_path):
    """

    :param model_path: path to hd5 of model
    :param weights_path: path to best weights file (checkpoint)
    :return: model instance to predict and evaluate
    """
    m= load_model(model_path)
    m.load_weights(weights_path)

    return m


def get_test_data(test_path):
    """
    gets data from folders into numpy arrays
    :param test_path: directory which contains the 3 folders (ends with /)
    :return: tuple (xd, yd, xe,ye,xl,yl)
    x's of shape(N,256,256,1)

    """
    path1=test_path+"diamond"
    path2=test_path+"ellipse"
    path3=test_path+"line"

    xd=load_folder(path1)
    xe=load_folder(path2)
    xl=load_folder(path3)

    yd=np.zeros((xd.shape[0],),dtype=np.uint8)
    ye=np.zeros((xe.shape[0],),dtype=np.uint8)+1
    yl=np.zeros((xl.shape[0],),dtype=np.uint8)+2

    print("shape diamond images : ", xd.shape)
    print("shape of ellipse images :  ", xe.shape)
    print("shape of line images :  ", xl.shape)

    return( xd,yd, xe,ye,xl,yl)
    pass


#load data from test file and appropriate model and return accuracy, save outputs in a file
def test_model(model,x,y,outpath):
    """

    :param model: loaded model
    :param x: images (N ,256,256,1)
    :param y:labels (N,)
    :param outpath: path to write classification results
    :return: -
    """

    cl=""
    if y[0]==0:
        cl="diamonds classification accuracy: "
    elif y[0]==1:
        cl="ellipse classification accuacy : "
    else:
        cl="line classification accuracy: "


    scores = model.evaluate(x,y,verbose=False)


    print("%s: %.2f%%" % (cl, scores[1]*100))

    predictions = model.predict(x,verbose=False)
    predicted_class=np.argmax(predictions,axis=1)
    diff=y-predicted_class
    print("indexes of images which were misclassified : ")
    print(np.nonzero(diff))
    save_output(x,predicted_class,outpath)
    return diff[diff!=0].shape[0]

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def main():
    #path to folder containg test folders
    #rename folder to "diamond, ellipse, line"
    #input_path="/media/mayar/not_fun/year4/pattern/image classifier/"
    input_path="/media/mayar/not_fun/year4/pattern/image classifier/test_data/"


    #path to folder which which will contain the classification results
    output_path="results/"

    #path to folder which contains model & weights
    model_path="/media/mayar/not_fun/year4/pattern/image classifier/neural_net experiments/model1/model.h5"
    weights_path="/media/mayar/not_fun/year4/pattern/image classifier/neural_net experiments/model1/experiments/exp13/checkpoints/weights-best-1.00.hdf5"
    model=model_load(model_path,weights_path)
    #don't forget to change png to bmp
    xd,yd,xe,ye,xl,yl=get_test_data(input_path)

    create_dir(output_path+"diamond_results/")
    create_dir(output_path+"ellipse_results/")
    create_dir(output_path+"line_results/")



    mis1=test_model(model, xd,yd, output_path+"diamond_results/")
    mis2=test_model(model, xe,ye, output_path+"ellipse_results/")
    mis3=test_model(model, xl,yl, output_path+"line_results/")
    total_mis= (mis1+mis2+mis3)/(yd.shape[0]+ ye.shape[0]+ yl.shape[0])

    print("total accuracy" ,(1-total_mis)*100)

main()
