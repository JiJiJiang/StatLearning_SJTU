#!/usr/bin/env python3.6
# coding:utf-8
# Author: Hongji Wang
# Email: jijijiang77@gmail.com

import numpy as np
import csv

def read_train(filename):
    X=[]
    Y=[]
    with open(filename) as fp:
        fp.readline() # ignore first line
        csv_reader = csv.reader(fp)
        for row in csv_reader:
            id = int(row[0])
            x = [float(value) for value in row[1:-1]]
            y = int(row[-1])
            X.append(x)
            Y.append(y)
    return (np.array(X),np.array(Y))

def read_test(filename):
    ids=[]
    X=[]
    with open(filename) as fp:
        fp.readline() # ignore first line
        csv_reader = csv.reader(fp)
        for row in csv_reader:
            id = int(row[0])
            x = [float(value) for value in row[1:]]
            ids.append(id)
            X.append(x)
    return (np.array(ids),np.array(X))

def writeResultCsv(ids,Y_pred,filename):
    result=np.column_stack((ids,Y_pred))
    with open(filename, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id','categories'])
        m = len(result)
        for i in range(m):
            writer.writerow(result[i])
            
            