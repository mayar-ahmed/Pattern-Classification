
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
def augment_data(cls):
    x,y,dir,x_dir,y_dir,xout,yout=None,None,"","","","",""
    if cls==0:
        dir="diamond"
        x_dir="data/xd.npy"
        xout="data/xd_aug.npy"

    elif cls==1:
        dir="ellipse"
        x_dir="data/xe.npy"
        xout="data/xe_aug.npy"

    else:
        dir="line"
        x_dir="data/xl.npy"
        xout="data/xl_aug.npy"

    x= np.load(x_dir)

    print ("x shape: ",x.shape )

    datagen = ImageDataGenerator(
        rotation_range=50,
        width_shift_range=0.3,
        height_shift_range=0.3,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

    i=0
    x_b=[]

    for x_batch in datagen.flow(x, batch_size=10,save_to_dir=dir, save_prefix='aug', save_format='png'):
        x_b.append(x_batch)
        i += 1
        if i > 210:
         break

    x_total=np.concatenate(x_b)
    print("x shape after stacking" , x_total.shape)
    finalx=np.concatenate((x,x_total),axis=0)

    finalx = finalx.astype(np.uint8)
    np.save(xout,finalx)
    print("final x" , finalx.shape , finalx.dtype)
    print("######################################")



augment_data(0)
augment_data(1)
augment_data(2)