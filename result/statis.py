#!/usr/bin/env python3.6
# coding:utf-8
# Author: Hongji Wang
# Email: jijijiang77@gmail.com

import csv
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, default="result.csv", help="")
    args = parser.parse_args()
    filename=args.filename
    with open(filename) as fp:
        fp.readline() # ignore first line
        csv_reader = csv.reader(fp)
        label2num=np.zeros(12)
        for row in csv_reader:
            #id = int(row[0])
            pred = int(row[-1])
            label2num[pred] += 1
    for label, num in enumerate(label2num):
        print("Label {}: number {}".format(label,num))
    print("Mean: {}, Std: {}".format(np.mean(label2num),np.std(label2num)))

if __name__ == "__main__":
    main()