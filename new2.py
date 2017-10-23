#coding: utf-8

############################
# import一些需要用到的库

import numpy as np
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn import svm
from sklearn import preprocessing
import csv
import pywt
import datetime

############################
# Method declaration，定义一些需要用到的方法函数

# read data from csv file，从csv文件读取数据
# 输入： filepath： csv文件路径， container： 存放数据的容器
def getDataFromCSV(filepath, container):
    with open(filepath, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            container.append(row)
        print "Reading data from " + filepath + "..."
        print "Records in total:"
        print len(container)


# dwt * 4（5） with filter db1， 使用pywt包中的dwt方法，对输入信号做若干次dwt变换
# 输入： signal： 输入信号
# 输出： res： dwt变换后的输出信号
# 此处做了3次dwt变换，共产生4对高低频段的结果
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
    # res.extend(cA6)
    # res.extend(cD6)
    res.extend(cA3)
    res.extend(cD3)
    res.extend(eA3)
    res.extend(eD3)
    res.extend(fA3)
    res.extend(fD3)
    res.extend(dA3)
    res.extend(dD3)
    # res.extend(dA6)
    # res.extend(dD6)
    return res

# dwt for a raw data matrix，对输入的样本矩阵做dwt变换
# 输入： matrix： 样本矩阵, featureNum: 属性个数
# 输出： outMatrix： 样本输出
def dwtForMatrix(matrix, featureNum):
    index = 0
    outMatrix = []
    singleSam = []
    # 对每一个样本
    for i in range(len(matrix)):
        # 中的每一个属性
        for j in range(featureNum):
            # 分别提取出每个属性，做dwt变换并把结果拼如singleSam
            singleSam.extend((dwt4time(matrix[i][j*50 : j*50 + 50])))
        # 把变换后的样本加入输出矩阵
        outMatrix.append(singleSam)
        singleSam = []
    return outMatrix

# generate a train-sample， 生成一个训练样本
# 输入： dataSource： 输入数据源
# 输出： oneSam： 输出样本
def getTrainSample(dataSource):
    oneSam = []
    # 对原始数据中第3-33列为原始行车数据属性
    for i in range(3, 33):
        dimList = []

        # skip some useless or not important information
        # 跳过11,21,27（行使里程数，发动机运行时间等相关性不大的属性），以及特征提取结果中产生非正影响的14个属性
        if i == 11 or i == 21  or i == 27 \
                or i == labelToInt('ENIDLESPDRDUCDRNG') \
                or i == labelToInt('AZ') \
                or i == labelToInt('BASETRGTENIDLESPD') \
                or i == labelToInt('DRVRDOOROPENSTS') \
                or i == labelToInt('TRIMMDCRKSHFTTOQREQVAL') \
                or i == labelToInt('THROTTLE_RELA_POSITION') \
                or i == labelToInt('CLPOS') \
                or i == labelToInt('ACCELERATORPADEL_POSITIONE') \
                or i == labelToInt('ACCELACTUPOS') \
                or i == labelToInt('ABSOLUTE_LOAD') \
                or i == labelToInt('ACCELERATORPADEL_POSITIOND') \
                or i == labelToInt('THROTTLE_ABS_POSITION') \
                or i == labelToInt('THROTTLE_ABS_POSITIONB') \
                or i == labelToInt('FIRSTCYLINDER_IAD') \
                :
            continue

        # 量化车门开闭
        if i == 20:
            for item in dataSource:
                if item[i] == 'closed':
                    dimList.append(1)
                else:
                    dimList.append(0)
            # dwt4result = dwt4time(dimList)
            dwt4result = dimList
        # 量化档位情况
        elif i == 13:
            for item in dataSource:
                dimList.append(gearToInt(item[i]))
            # dwt4result = dwt4time(dimList)
            dwt4result = dimList
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
        # 把每一维度的结果拼入样本输出
        oneSam.extend(dwt4result)
    return oneSam

# gear to int， 量化档位信息
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


# gear to int， 量化档位信息
# tansform the gear information to number
def gear2ToInt(gear):
    switcher = {
        'Reverse': -2,
        'Not in gear': -1,
        'Neutral': 0,
        'Forward gear': 1,
    }
    return switcher.get(gear, 0)

# judge whether the time is successive，判断时间上是否连续
# 输入： time1： 时间1， time2： 时间2
# 输出： boolean
def testSuccession(time1,time2):
    h1 = int(time1[0:2])
    m1 = int(time1[3:5])
    s1 = int(time1[6:8])
    ms1 = int(time1[9:11])

    h2 = int(time2[0:2])
    m2 = int(time2[3:5])
    s2 = int(time2[6:8])
    ms2 = int(time2[9:11])

    # 转换成毫秒判断
    total1 = h1*360000 + m1*6000 + s1*100 + ms1
    total2 = h2*360000 + m2*6000 + s2*100 + ms2
    if total2 - total1 == 2:
        return True
    else:
        return False

# get sample list， 获得训练样本集
# 输入：recordsPerSam： 每个样本的原始数据条目数， delay： 假定的数据延迟，
#      datasource： 输入数据源， sampleList： 训练样本集， classList： 样本对应的标签集
def getSampleListWithParams(recordsPerSam, delay, datasource, sampleList, classList):
    print "Start getting sampleList..."
    tmp = []
    delayIndex = delay * 50
    currentState = datasource[delayIndex][1]
    currentTime = datasource[delayIndex][0][11:22]
    count = 0
    index = 0
    for i in range(delayIndex, len(datasource)):
        # count only when in the same state and time successive， 仅当两条记录标签一致且时间连续，才计数
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

# 计算预测结果命中率
# 输入： a： 标签集1， b： 标签集2
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

# def getSubsetMultiDim(a,id1,id2):
#     b = a[:]
#     for i in range(len(b)):
#         for j in range(id2-id1):
#             b[i].pop(id1)
#     return b
#
# def getSubset(a,id1,id2):
#     b = a[:]
#     for j in range(id2-id1):
#         b.pop(id1)
#     return b

# 量化标签
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
# 代码主体部分

if __name__ == '__main__':

    startTime = datetime.datetime.now()

    delay = 0
    recordPerSam = 50

    # # init the data containers， 15 in total
    # dataContainer = []
    # filepath = '1222/day1206_1222.csv'
    #
    # # read data from csv file， 读取原始数据
    # getDataFromCSV(filepath, dataContainer)
    #
    # # print len(dataContainer)
    #
    # # 取出标题行
    # datasource = dataContainer[1:]

    dataContainer = []
    # dataContainer1 = []
    # dataContainer2 = []
    # dataContainer3 = []
    # dataContainer4 = []
    # dataContainer5 = []
    filepath = '1222/day1206_1222.csv'
    # filepath1 = '1222/day1205_1222.csv'
    # filepath2 = '1222/day1121_1222.csv'
    # filepath3 = '1222/day1123_1222.csv'
    # filepath4 = '1222/day1128_1222.csv'
    # filepath5 = '1222/day1129_1222.csv'

    # read data from csv file
    getDataFromCSV(filepath, dataContainer)
    # getDataFromCSV(filepath1, dataContainer1)
    # getDataFromCSV(filepath2, dataContainer2)
    # getDataFromCSV(filepath3, dataContainer3)
    # getDataFromCSV(filepath4, dataContainer4)
    # getDataFromCSV(filepath5, dataContainer5)

    # print len(dataContainer)

    datasource = dataContainer[1:]
                 # + dataContainer1[1:] \
                 # + dataContainer2[1:] \
                 # + dataContainer3[1:] \
                 # + dataContainer4[1:] \
                 # + dataContainer5[1:]

    print len(datasource)
    sampleList = []
    classList = []

    # 生成样本
    print "Generating training sample..."
    getSampleListWithParams(recordPerSam, delay, datasource, sampleList, classList)

    print "Raw sample:"
    print sampleList[0]
    print len(sampleList)
    print len(sampleList[0])

    # preprocessing scale， 标准化样本
    print "Preprocessing..."
    sampleList = preprocessing.scale(sampleList)
    print "Preprocessing done."
    print "Scaled sample:"
    print sampleList[0]
    print len(sampleList)
    print len(sampleList[0])

    # dwt processing， 对样本做dwt变换
    print "DWT..."
    sampleList = dwtForMatrix(sampleList, 13)
    print "DWT done."
    print "DWT sample:"
    print sampleList[0]
    print len(sampleList)
    print len(sampleList[0])

    print len(classList)
    # print classList[0]
    # print classList[500]
    # print classList[1000]

    print "The length of the sampleList:"
    print len(sampleList)
    print "The length of every single sample:"
    print len(sampleList[0])

    # 转换训练样本和标签集的容器
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

    print X
    print len(X)
    print Y
    print len(Y)

    # 生成svc，设置参数，判别方式为‘one-versus-one’，和函数为径向基函数，惩罚系数为10000.0
    clf = svm.SVC(decision_function_shape='ovo', kernel='rbf', C = 10000.0)
    accs = []
    round = 0

    # crossvalidate the scores on a number of different random splits of the data
    # 训练样本中随机洗牌10次，每次取0.3作为测试样本，其余作为训练样本
    for train_idx, test_idx in ShuffleSplit(len(X), 10, .3):
        round = round + 1
        print "Shuffle ", round, ": "
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        # 训练分类器
        print "Training..."
        r = clf.fit(X_train, Y_train)
        print "Training finished."

        # 计算本次训练预测结果
        acc = hitRate(Y_test, clf.predict(X_test))
        accs.append(acc)
        # print Y_test
        print "test sample: ", len(Y_test)
        print "Acc:", acc

    # 输出若干次洗牌的准确率结果
    print "Mean Accuracy: ", sum(accs) / len(accs)

    # 输出程序运行时间
    endTime = datetime.datetime.now()
    print "Running time:", (endTime - startTime).seconds, "s"


####################

# test 1
# all file + dwt&scale + 'rbf' + C = 10000.0 + 13 features
# Mean Accuracy:  0.745996407381

# test 2
# 1206 + dwt(highest-lowest) + c = 10000.0 + 13 features
# Mean Accuracy:  0.879985754986   3
# Mean Accuracy:  0.877991452991   4
# Mean Accuracy:  0.86655982906    2
# Mean Accuracy:  0.863603988604   5
# Mean Accuracy:  0.859152421652   6
# Mean Accuracy:  0.859472934473   6*2

# test 3
# 1206 + dwt + c = 10000.0 + 13 features
# Mean Accuracy:  0.854237891738


