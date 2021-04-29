#https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
# https://github.com/harrypnh/decision-tree-from-scratch
import random
import pandas
import time
from decisionTree import trainTestSplit, buildDecisionTree, decisionTreePredictions, calculateAccuracy

dataFrame = pandas.read_csv("WINE_DATA.csv")
# dataFrame = dataFrame.drop("id", axis = 1)
dataFrameTrain = dataFrame[dataFrame.columns.tolist()[1: ] + dataFrame.columns.tolist()[0: 1]]
# dataFrameTrain, dataFrameTest = trainTestSplit(dataFrame, testSize = )

print("Decision Tree - Predict Wine dataset")

i = 1
accuracyTrain = 0
# while accuracyTrain < 100:
#     startTime = time.time()
#     decisionTree = buildDecisionTree(dataFrameTrain, maxDepth = i)
#     buildingTime = time.time() - startTime
#     decisionTreeTestResults = decisionTreePredictions(dataFrameTest, decisionTree)
#     accuracyTest = calculateAccuracy(decisionTreeTestResults, dataFrameTest.iloc[:, -1]) * 100
#     decisionTreeTrainResults = decisionTreePredictions(dataFrameTrain, decisionTree)
#     accuracyTrain = calculateAccuracy(decisionTreeTrainResults, dataFrameTrain.iloc[:, -1]) * 100
#     print("maxDepth = {}: ".format(i), end = "")
#     print("accTest = {0:.2f}%, ".format(accuracyTest), end = "")
#     print("accTrain = {0:.2f}%, ".format(accuracyTrain), end = "")
#     print("buildTime = {0:.2f}s".format(buildingTime), end = "\n")
#     i += 1
TEST_USERS = [
    [3, 12.96, 3.45, 2.35, 18.5, 106, 1.39, .7, .4, .94, 5.28, .68, 1.75, 1.0, 675],
    [3, 13.17, 2.59, 2.37, 20, 120, 1.65, .68, .53, 1.46, 9.3, .6, 1.62, 1.0, 840],
    [1, 13.16, 2.36, 2.67, 18.6, 101, 2.8, 3.24, .3, 2.81, 5.68, 1.03, 3.17, 1.0, 1185],
    [2, 12.37, .94, 1.36, 10.6, 88, 1.98, .57, .28, .42, 1.95, 1.05, 1.82, 1.0, 520]
]

def Train():
    while accuracyTrain < 100:
        startTime = time.time()
        decisionTree = buildDecisionTree(dataFrameTrain, maxDepth = i)
        buildingTime = time.time() - startTime

def solution(WINE_DATA, test_user):
    # Your code goes here

    # while accuracyTrain < 100:
    #     startTime = time.time()
    #     decisionTree = buildDecisionTree(dataFrameTrain, maxDepth = i)
    #     buildingTime = time.time() - startTime
    #     decisionTreeTestResults = decisionTreePredictions(dataFrameTest, decisionTree)
    #     accuracyTest = calculateAccuracy(decisionTreeTestResults, dataFrameTest.iloc[:, -1]) * 100
    #     decisionTreeTrainResults = decisionTreePredictions(dataFrameTrain, decisionTree)
    #     accuracyTrain = calculateAccuracy(decisionTreeTrainResults, dataFrameTrain.iloc[:, -1]) * 100
    #     print("maxDepth = {}: ".format(i), end = "")
    #     print("accTest = {0:.2f}%, ".format(accuracyTest), end = "")
    #     print("accTrain = {0:.2f}%, ".format(accuracyTrain), end = "")
    #     print("buildTime = {0:.2f}s".format(buildingTime), end = "\n")
    #     i += 1

    # X = [3, 12.96, 3.45, 2.35, 18.5, 106, 1.39, .7, .4, .94, 5.28, .68, 1.75, 1.0, 675],

    return WINE_TYPE

dataFrameTest = [
    [3, 12.96, 3.45, 2.35, 18.5, 106, 1.39, .7, .4, .94, 5.28, .68, 1.75, 1.0, 675],
    [3, 13.17, 2.59, 2.37, 20, 120, 1.65, .68, .53, 1.46, 9.3, .6, 1.62, 1.0, 840],
    [1, 13.16, 2.36, 2.67, 18.6, 101, 2.8, 3.24, .3, 2.81, 5.68, 1.03, 3.17, 1.0, 1185],
    [2, 12.37, .94, 1.36, 10.6, 88, 1.98, .57, .28, .42, 1.95, 1.05, 1.82, 1.0, 520]
]
# solution(WINE_DATA, TEST_USER[1:])

for TEST_USER in TEST_USERS:
    WINE_TYPE = solution(WINE_DATA, TEST_USER[1:],)
    if WINE_TYPE == TEST_USER[0]:
        print(True)
    else:
        print(False)