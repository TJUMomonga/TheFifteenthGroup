#coding: utf-8
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import numpy as np
from scipy.stats import pearsonr
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.datasets import load_boston
from sklearn import svm
from sklearn import preprocessing
from sklearn.externals import joblib
import csv
import pywt
import datetime
import os
from sklearn.ensemble import RandomForestRegressor
# np.random.seed(0)
# size = 300
# x = np.random.normal(0, 1, size)
# print x
# print "Lower noise", pearsonr(x, x + np.random.normal(0, 1, size))
# print "Higher noise", pearsonr(x, x + np.random.normal(0, 10, size))
# boston = load_boston()
# X = boston["data"]
# Y = boston["target"]
# names = boston["feature_names"]
#
# print X
# print X[:,0:1]
# print X[:,1:2]
# print names

# read data from csv file
def getDataFromCSV(filepath, container):
    with open(filepath, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            container.append(row)
        print "Reading data from " + filepath + "..."
        print "Records in total:"
        print len(container)


# dwt * 4（5） with filter db1
def dwt4time(signal):
    (cA1, cD1) = pywt.dwt(signal, 'db1')
    # print cA1
    (cA2, cD2) = pywt.dwt(cA1, 'db1')
    (cA3, cD3) = pywt.dwt(cA2, 'db1')

    (eA3, eD3) = pywt.dwt(cD2, 'db1')
    # (cA4, cD4) = pywt.dwt(cA3, 'db1')
    # (cA5, cD5) = pywt.dwt(cA4, 'db1')
    # (cA6, cD6) = pywt.dwt(cA5, 'db1')
    # (cA7, cD7) = pywt.dwt(cA6, 'db1')

    (dA2, dD2) = pywt.dwt(cD1, 'db1')
    (dA3, dD3) = pywt.dwt(dD2, 'db1')

    (fA3, fD3) = pywt.dwt(dA2, 'db1')
    # (dA4, dD4) = pywt.dwt(dD3, 'db1')
    # (dA5, dD5) = pywt.dwt(dD4, 'db1')
    # (dA6, dD6) = pywt.dwt(dD5, 'db1')
    # (dA7, dD7) = pywt.dwt(dD6, 'db1')

    # add the high part to the tail of low part
    res = []
    # merge the low-high pass from low to high
    res.extend(cA3)
    res.extend(cD3)
    res.extend(eA3)
    res.extend(eD3)
    res.extend(fA3)
    res.extend(fD3)
    res.extend(dA3)
    res.extend(dD3)
    return res

# dwt for a raw data matrix
def dwtForMatrix(matrix):
    index = 0
    featureNum = 27
    outMatrix = []
    singleSam = []
    for i in range(len(matrix)):
        for j in range(featureNum):
            singleSam.extend((dwt4time(matrix[i][j*50 : j*50 + 50])))
        outMatrix.append(singleSam)
        singleSam = []
    return outMatrix

# generate a train-sample
def getTrainSample(dataSource):
    oneSam = []
    for i in range(3, 33):
        dimList = []

        # skip some useless or not important information
        if i == 11 or i == 21  or i == 27 :
            continue

        if i == 20:
            for item in dataSource:
                if item[i] == 'closed':
                    dimList.append(1)
                else:
                    dimList.append(0)
            # dwt4result = dwt4time(dimList)
            dwt4result = dimList
        elif i == 13:
            for item in dataSource:
                dimList.append(gearToInt(item[i]))
            # dwt4result = dwt4time(dimList)
            dwt4result = dimList
        # elif i == 11:
        #     for item in dataSource:
        #         dimList.append(gear2ToInt(item[i]))
        #     dwt4result = dwt4time(dimList)
        else:
            for item in dataSource:
                dimList.append(float(item[i]))
            # dwt4result = dwt4time(preprocessing.scale(dimList))
            dwt4result = dimList
        # add the average accelerating rate into sample
        # if i == 9:
        #     oneSam.append(sum(dimList) / len(dimList))

        # add the average speed into sample
        # if i == 17:
        #     oneSam.append(sum(dimList)/len(dimList))

        # print len(dimList)
        # standart processing

        # dwt4result = dwt4time(preprocessing.scale(dimList))
        # scale to (0,1)
        # dwt4result = dwt4time(scaleTo01(dimList))
        # print len(dwt4result)
        oneSam.extend(dwt4result)
    return oneSam

# gear to int
# tansform the gear information to number
def gearToInt(gear):
    switcher = {
        'Reverse Gear': -2,
        'Park Gear': -1,
        'Neutral Gear': 0,
        'First Gear': 1,
        'Second Gear': 2,
        'Third Gear': 3,
        'Fourth Gear': 4,
        'Fifth Gear': 5,
        'Sixth Gear': 6,
    }
    return switcher.get(gear, 0)


# gear to int
# tansform the gear information to number
def gear2ToInt(gear):
    switcher = {
        'Reverse': -2,
        'Not in gear': -1,
        'Neutral': 0,
        'Forward gear': 1,
    }
    return switcher.get(gear, 0)

# judge whether the time is successive
def testSuccession(time1,time2):
    h1 = int(time1[0:2])
    m1 = int(time1[3:5])
    s1 = int(time1[6:8])
    ms1 = int(time1[9:11])

    h2 = int(time2[0:2])
    m2 = int(time2[3:5])
    s2 = int(time2[6:8])
    ms2 = int(time2[9:11])

    total1 = h1*360000 + m1*6000 + s1*100 + ms1
    total2 = h2*360000 + m2*6000 + s2*100 + ms2
    if total2 - total1 == 2:
        return True
    else:
        return False

# scale dataset into (0,1)
def scaleTo01(dataset):
    maximum = max(dataset)
    minimum = min(dataset)
    for item in dataset:
        item = item - minimum
    if maximum != minimum:
        for item in dataset:
            item = item / (maximum - minimum)
    return dataset

# get sample list version 2
# recordsPerSam
# delay
# datasource
# sampleList
def getSampleListWithParams(recordsPerSam, delay, datasource, sampleList, classList, predictSample, predictList):
    print "Start getting sampleList..."
    tmp = []
    delayIndex = delay * 50
    currentState = datasource[delayIndex][1]
    currentTime = datasource[delayIndex][0][11:22]
    count = 0
    index = 0
    for i in range(delayIndex, len(datasource)):
        # count only when in the same state and time successive
        if currentState == datasource[i][1] and testSuccession(currentTime, datasource[i][0][11:22]):
            count = count + 1
        else:
            # print count
            count = 1
            tmp = []
        tmp.append(datasource[i])
        if count % recordsPerSam == 0:
            sampleList.append(getTrainSample(tmp))
            classList.append(currentState)
            count = 0
            tmp = []
        currentState = datasource[i][1]
        currentTime = datasource[i][0][11:22]
    print "Done."


def hitRate(a,b):
    hit = 0
    if len(a) == len(b):
        for i in range(len(a)):
            if a[i] == b[i]:
                hit = hit + 1
    else:
        print "Length not match."
    print 'hit:',hit
    return (hit + 0.0)/len(a)

def getSubsetMultiDim(a,id1,id2):
    b = a[:]
    for i in range(len(b)):
        for j in range(id2-id1):
            b[i].pop(id1)
    return b

def getSubset(a,id1,id2):
    b = a[:]
    for j in range(id2-id1):
        b.pop(id1)
    return b


def labelToInt(label):
    switcher = {
        'AY': 3,
        'AZ': 4,
        'BARPRESSABS': 5,
        'ENINTKAIRTEM': 6,
        'ENCLNTTEM': 7,
        'BRKPDLDRVRAPPDPRS_H1': 8,
        'VSELONGTACC_H1': 9,
        'TRIMMDCRKSHFTTOQREQVAL': 10,
        # 'TRENGDSTA': 11,
        'CLPOS': 12,
        'TRESTDGEAR': 13,
        'BASETRGTENIDLESPD': 14,
        'ENIDLESPDRDUCDRNG': 15,
        'STRGWHLANG': 16,
        'VEHSPDAVGDRVN_H1': 17,
        'FUELCSUMP': 18,
        'FUELLVLPCNT': 19,
        'DRVRDOOROPENSTS': 20,
        # 'VEHODO_H1': 21,
        'ENSPD': 22,
        'ACCELACTUPOS': 23,
        'ENGINE_WATER_TEMPERATURE': 24,
        'FIRSTCYLINDER_IAD': 25,
        'THROTTLE_ABS_POSITION': 26,
        # 'ENGINE_OPER_TIME': 27,
        'ABSOLUTE_LOAD': 28,
        'THROTTLE_RELA_POSITION': 29,
        'THROTTLE_ABS_POSITIONB': 30,
        'ACCELERATORPADEL_POSITIOND': 31,
        'ACCELERATORPADEL_POSITIONE': 32,
    }
    return switcher.get(label, 0)

###################################################

startTime = datetime.datetime.now()

delay = 0
recordPerSam = 50

predictSample = []
predictClass = []

# init the data containers， 15 in total
dataContainer = []
filepath = '1222/day1206_1222.csv'

# read data from csv file
getDataFromCSV(filepath, dataContainer)

# print len(dataContainer)

datasource = dataContainer[1:]
print len(datasource)
sampleList = []
classList = []

print "Generating training sample..."
getSampleListWithParams(recordPerSam, delay, datasource, sampleList, classList, predictSample, predictClass)

# preprocessing scale
print "Preprocessing..."
sampleList = preprocessing.scale(sampleList)
print "Preprocessing done."

# dwt processing
print "DWT..."
sampleList = dwtForMatrix(sampleList)
print "DWT done."

print len(classList)
# print classList[0]
# print classList[500]
# print classList[1000]

print "The length of the sampleList:"
print len(sampleList)
print "The length of every single sample:"
print len(sampleList[0])

print "The length of the testList:"
print len(predictSample)


X = np.array(sampleList)
Y = np.array(classList)
names = ['AY',
         'AZ',
         'BARPRESSABS',
         'ENINTKAIRTEM',
         'ENCLNTTEM',
         'BRKPDLDRVRAPPDPRS_H1',
         'VSELONGTACC_H1',
         'TRIMMDCRKSHFTTOQREQVAL',
         # 'TRENGDSTA',
         'CLPOS',
         'TRESTDGEAR',
         'BASETRGTENIDLESPD',
         'ENIDLESPDRDUCDRNG',
         'STRGWHLANG',
         'VEHSPDAVGDRVN_H1',
         'FUELCSUMP',
         'FUELLVLPCNT',
         'DRVRDOOROPENSTS',
         # 'VEHODO_H1',
         'ENSPD',
         'ACCELACTUPOS',
         'ENGINE_WATER_TEMPERATURE',
         'FIRSTCYLINDER_IAD',
         'THROTTLE_ABS_POSITION',
         # 'ENGINE_OPER_TIME',
         'ABSOLUTE_LOAD',
         'THROTTLE_RELA_POSITION',
         'THROTTLE_ABS_POSITIONB',
         'ACCELERATORPADEL_POSITIOND',
         'ACCELERATORPADEL_POSITIONE']

# rf = RandomForestRegressor(n_estimators=20, max_depth=5)

print X
print len(X)
print Y
print len(Y)

# rf = RandomForestRegressor()
clf = svm.SVC(decision_function_shape='ovo', kernel='rbf', C = 100.0)
accs = []
round = 0
# scores = basic.defaultdict(list)
#crossvalidate the scores on a number of different random splits of the data
for train_idx, test_idx in ShuffleSplit(len(X), 3, .3):
  round = round + 1
  print "Shuffle ", round, ": "
  X_train, X_test = X[train_idx], X[test_idx]
  Y_train, Y_test = Y[train_idx], Y[test_idx]
  # r = rf.fit(X_train, Y_train)
  # acc = r2_score(Y_test, rf.predict(X_test))
  print "Training..."
  r = clf.fit(X_train, Y_train)
  print "Training finished."
  # acc = r2_score(Y_test, clf.predict(X_test))

  acc = hitRate(Y_test, clf.predict(X_test))
  accs.append(acc)
  # print Y_test
  print "test sample: ",len(Y_test)
  # print clf.predict(X_test)
  # print len(clf.predict(X_test))
  print "Acc:", acc

print "Mean Accuracy: ", sum(accs)/len(accs)
endTime = datetime.datetime.now()
print "Running time:", (endTime - startTime).seconds, "s"

####################
# kernel test

# test 1
# single file + dwt + shuffle 3 at 0.3 + 'linear'
# Shuffle 1: Acc: 0.75641025641
# shuffle 2: Acc: 0.750712250712
# shuffle 3: Acc: 0.744301994302
# Mean Accuracy:  0.750474833808

# test 2
# single file + dwt + shuffle 3 at 0.3 + default kernel('rbf')
# Shuffle 1: Acc: 0.774216524217
# shuffle 2: Acc: 0.783475783476
# shuffle 3: Acc: 0.77386039886
# Mean Accuracy:  0.777184235518

# test 3
# single file + dwt + shuffle 3 at 0.3 + 'poly'
# Shuffle 1: Acc: 0.760327635328
# shuffle 2: Acc: 0.75462962963
# shuffle 3: Acc: 0.756054131054
# Mean Accuracy:  0.75700379867

# test 4
# single file + dwt + shuffle 3 at 0.3 + 'sigmoid'
# Shuffle 1: Acc: 0.434116809117
# shuffle 2: Acc: 0.464031339031
# shuffle 3: Acc: 0.448361823362
# Mean Accuracy:  0.44883665717

#################
# decision function shape

# test 5
# single file + dwt + shuffle 3 at 0.3 + default kernel('rbf') + 'ovr'
# Shuffle 1: Acc: 0.773148148148
# shuffle 2: Acc: 0.782407407407
# shuffle 3: Acc: 0.766381766382
# Mean Accuracy:  0.773979107312

#################
# C

# test 6
# single file + dwt + shuffle 3 at 0.3 + default kernel('rbf') + 'C = 2.0'
# Shuffle 1: Acc: 0.801282051282
# shuffle 2: Acc: 0.793091168091
# shuffle 3: Acc: 0.793091168091
# Mean Accuracy:  0.795821462488

# test 7
# single file + dwt + shuffle 3 at 0.3 + default kernel('rbf') + 'C = 0.5'
# Shuffle 1: Acc: 0.750712250712
# shuffle 2: Acc: 0.750712250712
# shuffle 3: Acc: 0.752849002849
# Mean Accuracy:  0.751424501425

# test 8
# single file + dwt + shuffle 3 at 0.3 + default kernel('rbf') + 'C = 10.0'
# Shuffle 1: Acc: 0.826566951567
# shuffle 2: Acc: 0.831908831909
# shuffle 3: Acc: 0.834045584046
# Mean Accuracy:  0.83084045584

# test 8
# single file + dwt + shuffle 3 at 0.3 + default kernel('rbf') + 'C = 100.0'
# Shuffle 1: Acc: 0.827991452991
# shuffle 2: Acc: 0.8400997151
# shuffle 3: Acc: 0.835113960114
# Mean Accuracy:  0.834401709402

