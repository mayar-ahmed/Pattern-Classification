from scipy import misc
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
#data has problems, some images are 256,256,3 not 256,256

def load128(path):
    img=misc.imread(path)
    plt.imshow(img)
    plt.show()
    size=(128,128)
    img2=misc.imresize(img,size,interp='nearest')
    plt.imshow(img2)
    plt.show()


def img2array(path):
    img = misc.imread(path)
    s=np.array(img)
    return s


def load_folder(folder_path):
    x=[]
    for path in glob.glob(folder_path+"/*.bmp"):
        s=img2array(path)
        if s.shape==(256,256,3):
            continue
        x.append(img2array(path))


    m=np.stack(x)
    return m

def load_data():
    #load images from each folder, class0=diamod, class1=ellipse, class2=line
    folder1='/media/mayar/not_fun/year4/pattern/image classifier/diamond/'
    folder2='/media/mayar/not_fun/year4/pattern/image classifier/ellipse/'
    folder3='/media/mayar/not_fun/year4/pattern/image classifier/line/'

    x_diamond=load_folder(folder1)
    x_elipse=load_folder(folder2)
    x_line=load_folder(folder3)


    print(x_diamond.shape)
    print(x_elipse.shape)
    print(x_line.shape)

    #save your numpy arrays into files (without augmentation)
    #make images (256,256,1) for conv nets
    xd= x_diamond.reshape(x_diamond.shape + (1,))
    xe= x_elipse.reshape(x_elipse.shape + (1,))
    xl= x_line.reshape(x_line.shape + (1,))
    print(xd.shape)
    print(xe.shape)
    print(xl.shape)

    np.save("data/xd.npy", xd)
    np.save("data/xe.npy" ,xe)
    np.save("data/xl.npy", xl)




load_data()
#load128("/media/mayar/not_fun/year4/pattern/image classifier/ellipse/ellipse91.bmp")
# img2array("/media/mayar/not_fun/year4/pattern/image classifier/ellipse/ellipse1.bmp")