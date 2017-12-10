
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

#take care augmentation takes 4 channel (n, w,h, depth)
def augment_data(cls):
    #takes training data from data folder and augments it
    #output :
    #xt_aug (Nnew, 256,256,1)
    #yt_aug (New, )

    x,y,dir,x_dir,y_dir,xout,yout=None,None,"","","","",""
    # if cls==0:
    #     dir="diamond"
    #     x_dir="data/xd.npy"
    #     xout="data/xd_aug.npy"
    #
    # elif cls==1:
    #     dir="ellipse"
    #     x_dir="data/xe.npy"
    #     xout="data/xe_aug.npy"
    #
    # else:
    #     dir="line"
    #     x_dir="data/xl.npy"
    #     xout="data/xl_aug.npy"

    # x= np.load(x_dir)


    x_train=np.load("data/x_train.npy")
    y_train=np.load("data/y_train.npy")

    #make training data (N,256,256,1)
    x_train= x_train.reshape(x_train.shape + (1,))
    print ("x train shape: ",x_train.shape )

    datagen = ImageDataGenerator(
        rotation_range=50,
        width_shift_range=0.3,
        height_shift_range=0.3,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

    dir="augmented_training_data"
    i=0
    x_b=[]
    y_b=[]
    for x_batch,y_batch in datagen.flow(x_train,y_train,shuffle=True,batch_size=10,save_to_dir=dir, save_prefix='aug', save_format='png'):
        x_b.append(x_batch)
        y_b.append(y_batch)
        i += 1
        if i > 300:
         break

    #output:
    #xt_aug (N,256,256,1)
    #yt_aug (N,)
    #concatenate batches of augmented data + original training data
    x_total=np.concatenate(x_b)
    print("x shape after stacking" , x_total.shape)
    finalx=np.concatenate((x_train,x_total),axis=0)
    finalx = finalx.astype(np.uint8)


    #concatentae batches of training labels+original labels
    y_total=np.concatenate(y_b)
    print("y shape after stacking" , y_total.shape)
    finaly=np.concatenate((y_train,y_total),axis=0)
    finaly = finaly.astype(np.uint8)

    #shuffle data:
    N= finalx.shape[0]
    ind_list = [i for i in range(N)]
    np.random.shuffle(ind_list)
    finalx = finalx[ind_list, :,:]
    finaly = finaly[ind_list,]

    np.save("data/xt_aug.npy",finalx)
    np.save("data/yt_aug.npy",finaly)


    print("final x" , finalx.shape , finalx.dtype)
    print("final y" , finaly.shape , finalx.dtype)
    print("######################################")

###################################################################
augment_data(0)



