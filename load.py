from scipy import misc
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
#data has problems, some images are 256,256,3 not 256,256

# def load128(path):
#     img=misc.imread(path)
#     plt.imshow(img)
#     plt.show()
#     size=(128,128)
#     img2=misc.imresize(img,size,interp='nearest')
#     plt.imshow(img2)
#     plt.show()


def img2array(path):
    #reads single image into numpy array
    img = misc.imread(path)
    s=np.array(img)
    return s


def load_folder(folder_path):
    #takes paths and returns its images as numpy array
    x=[]
    for path in glob.glob(folder_path+"/*.bmp"):
        s=img2array(path)
        if s.shape==(256,256,3):
            continue
        x.append(img2array(path))


    m=np.stack(x)
    return m

def load_data(path):
    #takes path of folders which contains the three folders of images
    #returns:
    # xtrain (N, 256,256) , ytrain(N,)
    # x_val (N,256,256) , y_val(N,)
    #both saved in data folder
    #load images from each folder, class0=diamod, class1=ellipse, class2=line

    #create directory data to save output
    if not os.path.exists("data/"):
        os.makedirs("data/")

    folder1=path+'diamond/'
    folder2=path+'ellipse/'
    folder3=path+'line/'

    xd=load_folder(folder1)
    xe=load_folder(folder2)
    xl=load_folder(folder3)


    print(xd.shape)
    print(xe.shape)
    print(xl.shape)

    #save ndarryas of each class to use later if needed
    # np.save("data/xd.npy", xd)
    # np.save("data/xe.npy" ,xe)
    # np.save("data/xl.npy", xl)
    ########################################################33
    #split into train, test, validation

    n1=xd.shape[0]
    n2=xe.shape[0]
    n3=xl.shape[0]


    #take 70% of each one for training
    b1 = int(0.7*n1)
    b2=int(0.7*n2)
    b3=int(0.7*n3)
    print("total train = ", b1+b2+b3)
    print("total val = " , n1-b1 +n2-b2+n3-b3)

    #no. of training points for each class
    #take random training data from each class
    mask1 = np.random.choice(n1, b1,replace=False)
    mask2=np.random.choice(n2, b2,replace=False)
    mask3=np.random.choice(n3,b3,replace=False)

    #70% of training data for each class
    train_diamond = xd[mask1]
    train_elipse=xe[mask2]
    train_line=xl[mask3]

    #rest is validation data
    val_diamond = np.delete(xd,mask1, axis=0)
    val_elipse = np.delete(xe,mask2 , axis=0)
    val_line = np.delete(xl,mask3,axis=0)

    #training labels
    ytd = np.zeros((b1,))
    yte=np.ones((b2,))
    ytl=np.ones((b3,))+1

    #validation labels
    yvd=np.zeros((n1-b1,))
    yve=np.ones((n2-b2,))
    yvl=np.ones((n3-b3,))+1

    #merge data from classes into x_train and val
    X_train=np.concatenate((train_diamond,train_elipse,train_line) , axis=0)
    y_train=np.concatenate((ytd,yte,ytl),axis=0)

    X_val= np.concatenate((val_diamond,val_elipse,val_line),axis=0)
    y_val=np.concatenate((yvd,yve,yvl),axis=0)

    #cast to integer (no need to take space of float32)
    y_train = y_train.astype(np.uint8)
    y_val = y_val.astype(np.uint8)


    #shuffle training data
    N1= X_train.shape[0]
    ind_list = [i for i in range(N1)]
    np.random.shuffle(ind_list)
    x_train = X_train[ind_list, :,:]
    y_train = y_train[ind_list,]

    #shuffle validation data
    N2= X_val.shape[0]
    ind_list = [i for i in range(N2)]
    np.random.shuffle(ind_list)
    x_val = X_val[ind_list, :,:]
    y_val = y_val[ind_list,]


    #save training and validation data
    np.save("data/x_train.npy",x_train)
    np.save("data/y_train.npy",y_train)
    np.save("data/x_val.npy",x_val)
    np.save("data/y_val.npy",y_val)

    print('Train data shape: ', x_train.shape, " dtype ", x_train.dtype)
    print('Train labels shape : ', y_train.shape, " dtype ", y_train.dtype)
    print('Validation data shape: ', x_val.shape, " dtype ", y_val.dtype)
    print('Validation labels shape: ', y_val.shape, " dtype ", y_val.dtype)




#call function with direcotry which contains the folders of images
#rename foders to diamond, ellipse and line

load_data("/media/mayar/not_fun/year4/pattern/image classifier/")
