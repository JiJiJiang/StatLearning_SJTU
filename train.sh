#!/bin/bash
# coding:utf-8
# Author: Hongji Wang
# Email: jijijiang77@gmail.com

### parameter config ###
train_file=data/train.csv
model_type=SVM  # {RC: RidgeClassifer, KNN:KNeighbors, GNB:Gaussian Naive-Bayes, LR:Logistic Regression, LDA:Linear Discriminant Analysis, SVM:Support Vector Machine, MLP:Multi-layer Perceptron, EL:Ensemble Learning}

python3 source_code/main.py --train-file ${train_file} \
                            --train-or-test train \
                            --model-type ${model_type}
