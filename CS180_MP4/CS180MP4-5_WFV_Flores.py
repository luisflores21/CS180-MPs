#40 test subjects, 10 pictures per subject ==> 40 classes
#Use 6 per subject for training

import os
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
import matplotlib.pyplot as plt

currdir = os.getcwd()
datadir = currdir + '/dataset'

#NEURAL NETWORK

#Forming the arrays for training and testing
y_train = []
y_test = []

x_train = cv2.imread(datadir + '/s1/1.pgm',-1)
x_train = cv2.resize(x_train,(int(x_train.shape[0]/2),int(x_train.shape[1]/2)))
x_train = np.reshape(x_train,-1)

x_test = cv2.imread(datadir + '/s1/7.pgm',-1)
x_test = cv2.resize(x_test,(int(x_test.shape[0]/2),int(x_test.shape[1]/2)))
x_test = np.reshape(x_test,-1)

y_train.append('s1')
y_test.append('s1')

for i in range(40):
    truenum = i+1
    picdir = 's' + str(truenum)
    imgdir = datadir + '/' + picdir
    if truenum != 1:
        img = cv2.imread(imgdir + '/1.pgm',-1)
        img = cv2.resize(img,(int(img.shape[0]/2),int(img.shape[1]/2)))
        img = np.reshape(img,-1)
        x_train = np.vstack((x_train,img))
        y_train.append(picdir)
        img = cv2.imread(imgdir + '/7.pgm',-1)
        img = cv2.resize(img,(int(img.shape[0]/2),int(img.shape[1]/2)))
        img = np.reshape(img,-1)
        x_test = np.vstack((x_test,img))
        y_test.append(picdir)
    img = cv2.imread(imgdir + '/2.pgm',-1)
    img = cv2.resize(img,(int(img.shape[0]/2),int(img.shape[1]/2)))
    img = np.reshape(img,-1)
    x_train = np.vstack((x_train,img))
    y_train.append(picdir)
    img = cv2.imread(imgdir + '/3.pgm',-1)
    img = cv2.resize(img,(int(img.shape[0]/2),int(img.shape[1]/2)))
    img = np.reshape(img,-1)
    x_train = np.vstack((x_train,img))
    y_train.append(picdir)
    img = cv2.imread(imgdir + '/4.pgm',-1)
    img = cv2.resize(img,(int(img.shape[0]/2),int(img.shape[1]/2)))
    img = np.reshape(img,-1)
    x_train = np.vstack((x_train,img))
    y_train.append(picdir)
    img = cv2.imread(imgdir + '/5.pgm',-1)
    img = cv2.resize(img,(int(img.shape[0]/2),int(img.shape[1]/2)))
    img = np.reshape(img,-1)
    x_train = np.vstack((x_train,img))
    y_train.append(picdir)
    img = cv2.imread(imgdir + '/6.pgm',-1)
    img = cv2.resize(img,(int(img.shape[0]/2),int(img.shape[1]/2)))
    img = np.reshape(img,-1)
    x_train = np.vstack((x_train,img))
    y_train.append(picdir)
    img = cv2.imread(imgdir + '/8.pgm',-1)
    img = cv2.resize(img,(int(img.shape[0]/2),int(img.shape[1]/2)))
    img = np.reshape(img,-1)
    x_test = np.vstack((x_test,img))
    y_test.append(picdir)
    img = cv2.imread(imgdir + '/9.pgm',-1)
    img = cv2.resize(img,(int(img.shape[0]/2),int(img.shape[1]/2)))
    img = np.reshape(img,-1)
    x_test = np.vstack((x_test,img))
    y_test.append(picdir)
    img = cv2.imread(imgdir + '/10.pgm',-1)
    img = cv2.resize(img,(int(img.shape[0]/2),int(img.shape[1]/2)))
    img = np.reshape(img,-1)
    x_test = np.vstack((x_test,img))
    y_test.append(picdir)

svm_train = x_train
svm_test = x_test

#Using MLPClassifier
print("MLP")
clf = MLPClassifier(hidden_layer_sizes = (1288), max_iter=100)
#Standardization
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
clf.fit(x_train,y_train)
scores = clf.score(x_test,y_test)
print("Two Layer Neural Net, Hidden Layer size = half of input node, test set:")
print(scores)
scores = clf.score(x_train,y_train)
print("Two Layer Neural Net, Hidden Layer size = half of input node, train set:")
print(scores)

#MLPClassifier with hidden layer size = incremented by 250
print("MLP INCREMENT")
mlp_x = []
mlp_y_test = []
mlp_y_train = []
i = 1288
while(i < 2576):
    mlp_x.append(i)
    clf = MLPClassifier(hidden_layer_sizes = (i), max_iter=100)
    clf.fit(x_train,y_train)
    scores = clf.score(x_test,y_test)
    mlp_y_test.append(scores)
    print("Two Layer Neural Net, test set:")
    print(scores)
    scores = clf.score(x_train,y_train)
    mlp_y_train.append(scores)
    print("Two Layer Neural Net, train set:")
    print(scores)
    i += 250
clf = MLPClassifier(hidden_layer_sizes = (2576), max_iter=100)
mlp_x.append(2576)
clf.fit(x_train,y_train)
scores = clf.score(x_test,y_test)
mlp_y_test.append(scores)
print("Two Layer Neural Net, Hidden Layer size = input node, test set:")
print(scores)
scores = clf.score(x_train,y_train)
mlp_y_train.append(scores)
print("Two Layer Neural Net, Hidden Layer size = input node, train set:")
print(scores)

#plot
plt.plot(mlp_x,mlp_y_test)
plt.xlabel('hidden nodes')
plt.ylabel('accuracy, test')
plt.show()
plt.plot(mlp_x,mlp_y_train)
plt.xlabel('hidden nodes')
plt.ylabel('accuracy, train')
plt.show()

#Using PCA
print("PCA")
pca = PCA(n_components=10)
train_eigen = pca.fit_transform(x_train)
test_eigen = pca.fit_transform(x_test)
clf.fit(train_eigen,y_train)
scores = clf.score(test_eigen,y_test)
print("Two Layer Neural Net, Hidden Layer size = input node, test set, pca")
print(scores)
scores = clf.score(train_eigen,y_train)
print("Two Layer Neural Net, Hidden Layer size = input node, train set, pca")
print(scores)

#Using SVM
print("SVM")
clf = svm.SVC()
clf.fit(svm_train,y_train)
scores = clf.score(svm_test, y_test)
print("Using SVM with default parameters, test set")
print(scores)
print("Using SVM with default paramerts, train set")
scores = clf.score(svm_train, y_train)
print(scores)

#Poly kernel degrees 1-5
print("POLY")
poly_x = []
poly_y_test = []
poly_y_train = []
for i in range(5):
    truenum = i+1
    print(truenum)
    poly_x.append(truenum)
    clf = svm.SVC(kernel='poly',degree=truenum)
    clf.fit(svm_train,y_train)
    scores = clf.score(svm_test, y_test)
    poly_y_test.append(scores)
    print("poly kernel, Using SVM with default parameters, test set")
    print(scores)
    print("poly kernel, Using SVM with default paramerts, train set")
    scores = clf.score(svm_train, y_train)
    poly_y_train.append(scores)
    print(scores)

#plot
plt.plot(poly_x,poly_y_test)
plt.xlabel('poly degree')
plt.ylabel('accuracy, test')
plt.show()
plt.plot(poly_x,poly_y_train)
plt.xlabel('poly degree')
plt.ylabel('accuracy, train')
plt.show()

print("RBF")
#rbf kernel, gamma 0.1 - 1
rbf_x = []
rbf_y_test = []
rbf_y_train = []
inc = 0.1
while inc <=1:
    print(inc)
    rbf_x.append(inc)
    clf = svm.SVC(kernel='rbf',gamma=inc)
    clf.fit(svm_train,y_train)
    scores = clf.score(svm_test, y_test)
    rbf_y_test.append(scores)
    print("poly kernel, Using SVM with default parameters, test set")
    print(scores)
    print("poly kernel, Using SVM with default paramerts, train set")
    scores = clf.score(svm_train, y_train)
    rbf_y_train.append(scores)
    print(scores)
    inc += 0.1

#plot
plt.plot(rbf_x,rbf_y_test)
plt.xlabel('rbf gamma')
plt.ylabel('accuracy, test')
plt.show()
plt.plot(rbf_x,rbf_y_train)
plt.xlabel('rbf gamma')
plt.ylabel('accuracy, train')
plt.show()
