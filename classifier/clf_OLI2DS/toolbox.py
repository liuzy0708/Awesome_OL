import numpy as np


def findCommonKeys(classifier, row):  # find the common keys of two dictionaries
    return (set(classifier.keys()) & set(row.keys()))


def findDifferentKeys(dict1, dict2):
    return (set(dict1.keys()) - set(dict2.keys()))


def subsetDictionary(dictionary, intersection):  # extract subset of key-val pairs if in
    return dict((value, dictionary[value]) for value in intersection)


def dict2NumpyArray(dictionary):
    return np.array([dictionary[key] for key in sorted(dictionary.keys())])


def dotDict(dict1, dict2):
    returnValue = 0
    for key in dict1:
        returnValue += dict1[key] * dict2[key]
    return returnValue


def NumpyArray2Dict(numpyArray, keys):
    return {k: v for k, v in zip(keys, numpyArray)}

