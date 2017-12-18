'''
Created on Dec 13, 2017

@author: Rania
'''
import HOG 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.svm import SVC
import printLC
import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.externals import joblib

def main():
   
    

    x_train=np.load("data/x_train.npy")
    y_train=np.load("data/y_train.npy")
    x_train_aug=np.load("data/xt_aug.npy")
    y_train_aug=np.load("data/yt_aug.npy")
    x_valid=np.load("data/x_val.npy")
    y_valid=np.load("data/y_val.npy")
    print("data loaded")
    currentDT = datetime.datetime.now()
    print (str(currentDT))
    #feature extraction
  
    #valid_features=HOG.hog(x_valid)
    #np.save("HOG_features_valid.npy",valid_features)
    valid_features=np.load("HOG_features_valid.npy")
    """
    
    #using ordinary data
    train_features=HOG.hog(x_train)
    
    print("features extracted")
    currentDT = datetime.datetime.now()
    print (str(currentDT))
    #model
    linear_svc = SVC(C=1,kernel='linear')
    print("model created")
    #rbf_svc = SVC(kernel='rbf')
    
    linear_svc.fit(train_features_aug, y_train_aug)
    print("features fitted")
    currentDT = datetime.datetime.now()
    print (str(currentDT))
    training_score=linear_svc.score(train_features, y_train)
    print("training Accuracy: {}").format(training_score.mean())
    #plotting the learning curve
    title = "Learning Curves (svm, linear kernel, c=1) training set"
    # SVC is more expensive so we do a lower number of CV iterations:
    printLC.plot_learning_curve(linear_svc, title, train_features, y_train, (0.7, 1.01), n_jobs=4)
    plt.show()
    
    
    """
    #using augmented data
    """
    x_ta=np.reshape(x_train_aug, (3162,256,256))
    x_train_aug=x_ta
    train_features_aug=HOG.hog(x_train_aug)
    np.save("HOG_augmented_features_train.npy",train_features_aug)
    """
    train_features_aug=np.load("HOG_augmented_features_train.npy")
    currentDT = datetime.datetime.now()
    print (str(currentDT))
    #train_features_aug= np.load("feat_file.npy")
    print("features extracted")
    currentDT = datetime.datetime.now()
    print (str(currentDT))
    #model
    #linear_svc = SVC(C=5,kernel='linear')
    print("model created")
    #000000289
    rbf_svc = SVC(kernel='rbf',C=10,gamma='auto',decision_function_shape='ovr')
    
    rbf_svc.fit(train_features_aug, y_train_aug)
    print("features fitted")
    currentDT = datetime.datetime.now()
    print (str(currentDT))
        
    predicted = rbf_svc.predict(valid_features)
    # get the accuracy
    print("validation score: {}".format(accuracy_score(y_valid, predicted)))
    currentDT = datetime.datetime.now()
    print (str(currentDT))
    predicted = rbf_svc.predict(train_features_aug)
    # get the accuracy
    print("training score: {}".format(accuracy_score(y_train_aug, predicted)))
    joblib.dump(rbf_svc, "SVM_model_hog")
    
    """
    #plotting the learning curve
    title = "Learning Curves  training set"
    # SVC is more expensive so we do a lower number of CV iterations:
    
    printLC.plot_learning_curve(rbf_svc, title, train_features_aug, y_train_aug, n_jobs=-1,cv=10)
    plt.show()
    """


    
if __name__ == "__main__": main()
