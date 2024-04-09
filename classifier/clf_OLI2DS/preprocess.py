import csv
import random
from sklearn import preprocessing
import numpy as np
import math
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def readCreditA():
    dataset = []
    with open("./dataset/credit-a.csv") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset).astype(np.float)
    dataset[:, :-1] = preprocessing.scale(dataset[:, :-1])
    return dataset


def readKrVsKp():
    dataset = []
    with open("./otherDatasets/kr-vs-kp.csv") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset).astype(np.float)
    dataset[:, :-1] = preprocessing.scale(dataset[:, :-1])
    return dataset

def readSplice():
    dataset = []
    with open("./dataset/splice.csv") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset).astype(np.float)
    dataset[:, :-1] = preprocessing.scale(dataset[:, :-1])
    return dataset


def readSvmguide3Normalized():
    dataset = []
    with open("./otherDatasets/svmguide3.csv") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset).astype(np.float)
    dataset[:, :-1] = preprocessing.scale(dataset[:, :-1])
    return dataset


def readSpambaseNormalized():
    dataset = []
    with open("./dataset/spambase.csv") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset).astype(np.float)
    dataset[:, :-1] = preprocessing.scale(dataset[:, :-1])
    return dataset


def readWdbcNormalized():
    dataset = []
    with open("./dataset/wdbc.csv") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset).astype(np.float)
    dataset[:, :-1] = preprocessing.scale(dataset[:, :-1])
    return dataset

def readHyperplane2Normalized():
    dataset = []
    with open("./dataset/Hyperplane2.csv") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset).astype(np.float)
    dataset[:, :-1] = preprocessing.scale(dataset[:, :-1])
    return dataset

def NumpyArrary2Dict(dataset):
    dictData = [{k: v for k, v in enumerate(row)} for row in dataset]
    for row in dictData:
        row["class_label"] = row.pop(len(row)-1)
    return dictData


def removeRandomData(data):
    maxFeatureNum = len(data[0])
    featureKept = math.ceil(len(data[0]) * 0.5)
    chunkSize = math.ceil(len(data) * 0.5)
    howSparseLength = featureKept
    for (i, vec) in enumerate(data):
        if (i+1) % chunkSize == 0:
            howSparseLength = min(howSparseLength + featureKept, maxFeatureNum)
        rDelSamples = random.sample(range(maxFeatureNum), howSparseLength)
        for k, v in vec.copy().items():
            if k not in rDelSamples:
                if k!="class_label":
                    vec[k] = 0
                    # del vec[k]
    return data


def removeDataTrapezoidal(original_dataset):  # trapezoidal
    dataset = original_dataset[:]
    features = len(dataset[0])
    rows = len(dataset)
    for i in range(0, len(dataset)):
        multiplier = int(i / (rows / 10)) + 1
        increment = int(features / 10)
        features_left = multiplier * increment
        if i == len(dataset) - 1:
            features_left = features - 2
        for key, value in dataset[i].copy().items():
            if key != 'class_label' and key > features_left:
                # dataset[i][key]=0
                dataset[i].pop(key)
    return dataset

