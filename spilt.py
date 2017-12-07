import numpy as np
import matplotlib.pyplot as plt
xd = np.load("data/xd_aug.npy")
xe=np.load("data/xe_aug.npy")
xl=np.load("data/xl_aug.npy")
print(xd.dtype, xe.dtype, xl.dtype)

#number of datapoints in each class
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
#take training data from each class
mask1 = np.random.choice(n1, b1,replace=False)
mask2=np.random.choice(n2, b2,replace=False)
mask3=np.random.choice(n3,b3,replace=False)

train_diamond = xd[mask1]
train_elipse=xe[mask2]
train_line=xl[mask3]

val_diamond = np.delete(xd,mask1, axis=0)
val_elipse = np.delete(xe,mask2 , axis=0)
val_line = np.delete(xl,mask3,axis=0)

###train data
ytd = np.zeros((b1,))
yte=np.ones((b2,))
ytl=np.ones((b3,))+1

yvd=np.zeros((n1-b1,))
yve=np.ones((n2-b2,))
yvl=np.ones((n3-b3,))+1

X_train=np.concatenate((train_diamond,train_elipse,train_line) , axis=0)
y_train=np.concatenate((ytd,yte,ytl),axis=0)

X_val= np.concatenate((val_diamond,val_elipse,val_line),axis=0)
y_val=np.concatenate((yvd,yve,yvl),axis=0)

y_train = y_train.astype(np.uint8)
y_val = y_val.astype(np.uint8)


np.save("data/x_train.npy",X_train)
np.save("data/y_train.npy",y_train)
np.save("data/x_val.npy",X_val)
np.save("data/y_val.npy",y_val)

print('Train data shape: ', X_train.shape, " dtype ", X_train.dtype)
print('Train labels shape : ', y_train.shape, " dtype ", y_train.dtype)
print('Validation data shape: ', X_val.shape, " dtype ", X_val.dtype)
print('Validation labels shape: ', y_val.shape, " dtype ", y_val.dtype)

# rand= np.random.choice(X_train.shape[0],1)
# images=X_train[rand]
# labels=y_train[rand]
# print(rand[0])
# plt.imshow(images[0].reshape(256,256))
# plt.show()
# print(labels[0])
