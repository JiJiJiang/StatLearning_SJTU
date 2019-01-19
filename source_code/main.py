#!/usr/bin/env python3.6
# coding:utf-8
# Author: Hongji Wang
# Email: jijijiang77@gmail.com

import argparse
from sklearn import preprocessing
from sklearn.utils import shuffle
from DataIO import read_train, read_test, writeResultCsv
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.decomposition import PCA
### Ridge Classifier
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
### KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
### Gaussian Naive-Bayes
from sklearn.naive_bayes import GaussianNB
### Logistic Regression
from sklearn.linear_model import LogisticRegression
### Quadratic Discriminant Analysis, Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
### SVM Classifier    
from sklearn.svm import SVC, LinearSVC
### MLP Classifier
from sklearn.neural_network import MLPClassifier
### Ensemble
from sklearn.ensemble import BaggingClassifier, VotingClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, default="data/train.csv", help="")
    parser.add_argument('--test-file', type=str, default="data/test.csv", help="")
    parser.add_argument('--model-type', type=str, default="SVM", help="")
    parser.add_argument('--train-or-test', type=str, default="train", help="")
    parser.add_argument('--result-file', type=str, default="result/result.csv", help="")
    args = parser.parse_args()
    train_file=args.train_file
    test_file=args.test_file
    model_type=args.model_type
    train_or_test=args.train_or_test
    result_file=args.result_file

    print("Loading training data...")
    X, Y = read_train(train_file)
    X = preprocessing.scale(X)
    #pca = PCA(n_components=2048)
    #pca.fit(X)
    #print(pca.explained_variance_ratio_)
    #X = pca.transform(X)
    print("Finish loading training data!")
    # model_type = {RC: RidgeClassifer, KNN:KNeighbors, GNB:Gaussian Naive-Bayes, LR:Logistic Regression, LDA:Linear Discriminant Analysis, SVM:Support Vector Machine, MLP:Multi-layer Perceptron, EL:Ensemble Learning}
    if model_type == 'RC':
        ALPHA=10                                                                #50:67.32, 40:66.02, 30:64.71, 25:64.38, 20:64.43 10:66.35, 1:72.49
        print("alpha: {}".format(ALPHA))
        model = RidgeClassifier(alpha=ALPHA, normalize=True)                     #0.10,0,11:0.986923, 0.12,0.13,0.14:0.987179, 0.15:0.987051
    elif model_type == 'KNN':
        model = KNeighborsClassifier(n_neighbors=12, n_jobs=4)                      #0.975000,
    elif model_type == 'GNB':
        model = GaussianNB()                                                        #0.925897,
    elif model_type == 'LR':
        C=0.0005
        print("C: {}".format(C))
        model = LogisticRegression(C=C)                                             #1.0:0.9923077
    elif model_type == 'LDA':
        model = LinearDiscriminantAnalysis() #QuadraticDiscriminantAnalysis()       #0.978077
    elif model_type == 'SVM':
        #model = SVC(C=3.0, kernel='rbf', gamma='auto')                              #0.985890,0.987180,0.987436
        C=1e-4
        print("C: {}".format(C))
        model = LinearSVC(C=C)                                                  #0.001:0.988590,0.00075:0.988718
    elif model_type == 'MLP':
        model = MLPClassifier(random_state=1, max_iter=500, tol=1e-4, hidden_layer_sizes=(256,256), activation='relu',
                         solver='adam', alpha=1e-4, batch_size=256, learning_rate_init=0.0005, learning_rate='adaptive')  #0.986667,0.987051
    else: # EL:Ensemble Learning
        model = VotingClassifier( estimators=[("LR", LogisticRegression(C=0.001)), ("RC", RidgeClassifier(alpha=10, normalize=True)), ("SVM", LinearSVC(C=1e-4))],
                 voting="hard", n_jobs=-1)

    '''
    X1, Y1 = shuffle(X, Y, random_state=1)
    N=780
    X_train, X_dev = X1[N:], X1[:N]
    Y_train, Y_dev = Y1[N:], Y1[:N]
    model.fit(X_train, Y_train)
    acc = model.score(X_dev,Y_dev)
    print("Accuracy: {}".format(acc))
    #'''
    
    if train_or_test == 'train':
        k=10 #kFold
        skf=StratifiedKFold(n_splits=k, random_state=1, shuffle=True)
        #skf=KFold(n_splits=k)
        skf.get_n_splits(X,Y)
        print(skf)
        sum_acc = 0.0
        fold = 1
        for train_index, dev_index in skf.split(X,Y):
            #print("Train Index:", train_index, ",dev Index:", dev_index)
            X_train, X_dev = X[train_index], X[dev_index]
            Y_train, Y_dev = Y[train_index], Y[dev_index]
            model.fit(X_train, Y_train)
            acc = model.score(X_dev,Y_dev)
            sum_acc += acc
            print("Fold {} accuracy: {}".format(fold, acc))
            fold += 1
        average_acc = sum_acc/k
        print("Average accuracy: {}".format(average_acc))
    else:
        print("Training...")
        model.fit(X, Y)
        print("Finish training. Now testing...")
        ids, X_test = read_test(test_file)
        X_test = preprocessing.scale(X_test)
        #X_test = pca.transform(X_test)
        Y_pred = model.predict(X_test)
        writeResultCsv(ids,Y_pred,result_file)
        print("Finish testing!")


if __name__ == "__main__":
    main()