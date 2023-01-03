# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 18:42:23 2022

@author: jstei
"""

# * JakeSteinfinkP1.py
# *
# *  ANLY 555 2022
# *  Project Deliverable 1
# *
# *  Due on: 2/13/2022
# *  Author(s): Jake Steinfink
# *
# *
# *  In accordance with the class policies and Georgetown's
# *  Honor Code, I certify that, with the exception of the
# *  class resources and those items noted below, I have neither
# *  given nor received any assistance on this project other than
# *  the TAs, professor, textbook and teammates.
# *


# =====================================================================
# Testing script for Deliverable 1: Source Code Framework
# =====================================================================

# =====================================================================
# Testing DataSet Class
# (Not meant to be called, but will show instantiation, attributes,
# and member methods)
# =====================================================================
from DataSetClass import (
    DataSet,
    QuantDataSet,
    QualDataSet,
    TextDataSet,
    TimeSeriesDataSet,
    HeterogeneousData
)

TimeSeriesFile = "data/BTCUSD_day.csv"
TimeSeriesColumns = ["Date", "Open"]
QuantDataFile = "data/Sales_Transactions_Dataset_Weekly.csv"
QuantDataColumns = ["W0", "W1"]
TextDataFile = "data/yelp.csv"
QualDataFile = "data/multiple_choice_responses.csv"
QualDataColumns_mode = ["Q5", "Q6"]
QualDataColumns_median = ["Q5_OTHER_TEXT", "Q14_Part_1_TEXT"]


def DataSetTests():
    print("DataSet Instantiation invokes __readFromCSV() method....")
    data = DataSet(TimeSeriesFile)
    print("==============================================================")


def QuantDataSetTests():
    print(
        "QuantDataSet Instantiation invokes both the __load() and the\
__readFromCSV() methods...."
    )
    data = QuantDataSet(QuantDataFile)
    print("===========================================================")
    print("QuantDataSet.clean() with Columns parameter:")
    data.clean(Columns=QuantDataColumns)
    print("QuantDataSet.clean() with default parameter:")
    data.clean(Columns=QuantDataColumns)
    print("QuantDataSet.explore() with mandatory Columns parameter:")
    data.explore(Columns=QuantDataColumns)
    print("\n\n")


def QualDataSetTests():
    print(
        "QualDataSet Instantiation invokes both the __load() and the\
__readFromCSV() methods...."
    )
    data = QualDataSet(QualDataFile)
    print("===========================================================")
    print(
        "Check that clean method works with mode. No columns entered so all should be cleaned."
    )
    data.clean()
    print("Visualizing two columns")
    x = data.explore(Columns=QualDataColumns_mode)
    print("===========================================================")
    print("Reinstatiating and repeating process with median filling")
    data = QualDataSet(QualDataFile)
    print("Check that the clean method works with median")
    data.clean(Columns=QualDataColumns_median, value="median")
    print("Check that explore method works with defaults...\n")
    x = data.explore()
    print("===========================================================")
    print("\n\n")


def TextDataSetTests():
    print("QualDataSet Instantiation invokes the __load method....")
    data = TextDataSet(TextDataFile, ".txt")
    print("Check member methods work ...")
    print("TextDataSet.clean() with optional Stopwords and language parameters:")
    data.clean(Stopwords=["to"], language="english")
    print("TextDataSet.clean() with default parameters:")
    data.clean()
    print("TextDataSet.explore():")
    data.explore()
    print("\n\n")


def TimeSeriesDataSetTests():
    print("QualDataSet Instantiation invokes the __readFromCSV() method....")
    data = TimeSeriesDataSet(TimeSeriesFile)
    print("Check member methods work..")
    print("Check TimeSeriesDataSet member methods...")
    print(
        "TimeSeriesDataSet.clean() with optional parameter for Columns and default filter_size"
    )
    data.clean(Columns=["Open", "High"])
    print("TimeSeriesDataSet.clean() with optional parameters for filter_size:")
    data.clean(filter_size=1)
    print("TimeSeriesDataSet.explore() with mandatory Columns parameter:")
    data.explore(Columns=TimeSeriesColumns)
    print("\n\n")

def HeterogeneousDataSetTests():
    print("HeterogeneousDataSet Instantiation....")
    Quant = QuantDataSet(QuantDataFile)
    Qual = QualDataSet(QualDataFile)
    Text = TextDataSet(TextDataFile, ".txt")
    df_list = {Quant : QuantDataColumns , Qual : QualDataColumns_mode ,Text:''}
    data = HeterogeneousData(df_list)
    print("Check member methods work..")
    print("Check that clean method works on all...")
    data.clean_all()
    print("Check that explore method works on all...")
    data.explore_all()
    quant = data.select(0)
    print("\n\n")


# =====================================================================
# Testing Classifier Class
# (Not meant to be called, but will show instantiation, attributes,
# and member methods)
# =====================================================================
from ClassifierAlgorithmClass import (
    ClassifierAlgorithm,
    simpleKNNClassifier,
    DecisionTreeClassifier,
)
import random as rd

iris = "data/Iris.csv"


def ClassifierAlgorithmTests():
    print("ClassifierAlgorithm Instantiation....")
    classifier = ClassifierAlgorithm()
    print("Rest of methods will be tested below")
    print("===========================================================\n\n")


def simpleKNNClassifierTests():
    print("simpleKNNClassifier Instantiation....")
    classifier = simpleKNNClassifier()
    data = DataSet(iris)
    data.data.drop(["Id"], inplace=True, axis=1)
    train_indexes = rd.sample(
        [i for i in range(len(data.data))], int(0.75 * len(data.data))
    )
    traindata = data.data.iloc[train_indexes]
    trainlabels = traindata["Species"]
    traindata.drop(["Species"], inplace=True, axis=1)
    testdata = data.data.iloc[
        [i for i in range(len(data.data)) if i not in train_indexes]
    ]
    testlabels = testdata["Species"]
    testdata.drop(["Species"], inplace=True, axis=1)
    print("==============================================================")
    print("Check member member methods...")
    print("simpleKNNClassifierAlgorithm.train():")
    classifier.train(traindata, trainlabels)
    print("simpleKNNClassifier.test():")
    Predicted_Labels = classifier.test(testdata, testlabels)
    print("===========================================================\n\n")


def DecisionTreeClassifierTests():
    print("kdTreeKNNClassifier Instantiation....")
    classifier = DecisionTreeClassifier()
    data = DataSet(iris)
    data.data.drop(["Id"], inplace=True, axis=1)
    train_indexes = rd.sample(
        [i for i in range(len(data.data))], int(0.75 * len(data.data))
    )
    traindata = data.data.iloc[train_indexes]
    trainlabels = traindata["Species"]
    traindata.drop(["Species"], inplace=True, axis=1)
    testdata = data.data.iloc[
        [i for i in range(len(data.data)) if i not in train_indexes]
    ]
    testlabels = testdata["Species"]
    testdata.drop(["Species"], inplace=True, axis=1)
    print("==============================================================")
    print("Check member methods...")
    print("kdTreeKNNClassifier.train():")
    classifier.train(traindata, trainlabels)
    print("kdTreeKNNClassifier.test():")
    classifier.test(testdata)
    print(classifier.T)
    print("===========================================================\n\n")


# =====================================================================
# Testing Classifier Class
# (Not meant to be called, but will show instantiation, attributes,
# and member methods)
# =====================================================================
from ExperimentClass import Experiment




def ExperimentTests():
    print("Experiment class instantiation (Experiment())...")
    data = DataSet(iris)
    data.data.drop(["Id"], axis=1, inplace=True)
    randomized_data = data.data.sample(frac=1)
    experiment = Experiment(
        randomized_data, ["simpleKNNClassifier","DecisionTreeClassifier"], "Species"
    )
    print("==============================================================")
    print("Check class member methods...\n")
    print("Experiment.runCrossVal(numFolds):")
    experiment.runCrossVal(10)
    print("==============================================================")
    print("Experiment.score():")
    experiment.score()
    print("==============================================================")
    print("Experiment.confusionMatrix():")
    experiment.confusionMatrix()
    print("Experiment.ROC():")
    experiment.ROC()


def main():
    DataSetTests()
    QuantDataSetTests()
    QualDataSetTests()
    TextDataSetTests()
    TimeSeriesDataSetTests()
    ClassifierAlgorithmTests()
    simpleKNNClassifierTests()
    DecisionTreeClassifierTests()
    ExperimentTests()


if __name__ == "__main__":
    main()
