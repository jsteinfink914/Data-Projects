# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 12:40:03 2022

@author: jstei
"""

from ClassifierAlgorithmClass import (
    simpleKNNClassifier,
    DecisionTreeClassifier,
)
from tabulate import tabulate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Experiment:
    def __init__(self, Dataset, ClassifierList, labelColumn):
        """
         Parameters
         --------------
         Dataset: a dataset object with labels
         ClassifierList: a list of classifiers that should be run
         labelColumn: Column name that holds the labels

        Returns as attribute
        -------
        self.data: Dataset object
        self.ClassifierListL: list of classifiers to be applied
        self.labelColumn=label column name

        """
        self.data = Dataset
        self.ClassiferList = ClassifierList
        self.labelColumn = labelColumn

    def runCrossVal(self, k):
        """
        Makes use of ClassifierAlgorithmClass functions to run algorithms on
        k folds


        Parameters
        ----------
        k : how many cross validation sessions will be run

        Returns as attribute
        -------
        self.Predicted_Labels: Predicted Labels as a matrix of size NumClassifiers*numFolds
        self.True_Labels: Actual labels as a matrix of size NumClassifiers*numFolds

        """
        ##This list will be updated as more algorithms come online
        Classifiers_with_functionality = [
            "simpleKNNClassifier",
            "DecisionTreeClassifier",
        ]

        ##Filtering ClassifierList to only include ones with functionality
        self.ClassifierList = [
            i for i in self.ClassiferList if i in Classifiers_with_functionality
        ]

        ##Predicted labels will be stored in a matrix of size NumClassifiers*numfolds
        Predicted_Labels = [[0] * k for i in range(len(self.ClassifierList))]
        ##Actual labels will be stored in a matrix of size Numclassifiers*numfolds
        True_Labels = [[0] * k for i in range(len(self.ClassifierList))]

        ##For each fold
        for i in range(k):
            ##Calculate fold size
            fold_size = len(self.data) // k
            ##Test indices are a changing list of indices of size fold_size
            test_indexes = [
                i for i in range(0 + i * fold_size, fold_size + i * fold_size)
            ]
            ##Train indices are all other indices
            train_indexes = [i for i in range(len(self.data)) if i not in test_indexes]
            ##Select train and test data and isolate label columns
            train_data = self.data.iloc[train_indexes]
            train_labels = train_data[self.labelColumn]
            test_data = self.data.iloc[test_indexes]
            test_labels = test_data[self.labelColumn]
            test_data.drop([self.labelColumn], inplace=True, axis=1)
            train_data.drop([self.labelColumn], inplace=True, axis=1)

            ##For each classifier
            for a in range(len(self.ClassiferList)):
                ##Will become a series of elifs for each different classifier
                if self.ClassifierList[a] == "simpleKNNClassifier":
                    ##Initiate simpleKNNClassifer class
                    KNN = simpleKNNClassifier()
                    ##Feed train data and labels
                    KNN.train(train_data, train_labels)
                    ##Receive predicted labels
                    Labels = KNN.test(test_data, test_labels)
                    ##Store them in the matrix for the specific classifier and specific
                    ##fold
                    Predicted_Labels[a][i] = Labels
                    ##Store actual labels in same spot
                    True_Labels[a][i] = test_labels
                elif self.ClassifierList[a] == "DecisionTreeClassifier":
                    ##Initiate DecisionTreeClassifier
                    Tree = DecisionTreeClassifier()
                    ##Feed train data and labels
                    Tree.train(train_data, train_labels)
                    ##Get predicted labels
                    Labels = Tree.test(test_data)
                    ###Store predicted and actual labels
                    Predicted_Labels[a][i] = Labels
                    True_Labels[a][i] = test_labels

                ##If not a valid algorithm...
                else:
                    raise ValueError("Functionality has not been built yet.")
        ##Make predicted and actual labels attributes
        self.Predicted_Labels = Predicted_Labels
        self.True_Labels = True_Labels

    def score(self):
        """
        Computes the accuracy of the cross-validated algorithms for all
        algorithms being testes

        Prints to console
        -------
        Accuracy score based for each algorithm that is run.

        """
        ##Array to store results
        table = [[0] * 2 for i in range(len(self.ClassifierList))]
        ##For each classifier...
        for i in range(len(self.ClassifierList)):
            ##Empty list to hold accuracy calculations
            accuracy = []
            ##For each fold
            for a in range(len(self.True_Labels[0])):
                ##Store predicted and actual labels
                predicted_labels = self.Predicted_Labels[i][a]
                true_labels = self.True_Labels[i][a]
                ##Convert set of predicted labels into list of most likely label
                predicted_labels = [max(z,key=z.get) for z in predicted_labels]
                ##Append accuracy percentage for each fold
                ##Calculated by summing how many labels are directly equal to eachother
                ##and dividing by length of the list

                accuracy.append(
                    sum(1 for x, y in zip(true_labels, predicted_labels) if x == y)
                    / len(true_labels)
                )
            ##Overall score equals average of the accuracy scores for each fold
            ##Converted to a percent for the reader
            score = str(round((sum(accuracy) / len(accuracy)) * 100, 2)) + "%"
            ##Enter classifer name and score for the classifier
            table[i] = [self.ClassifierList[i], score]
        ##Column names for the table
        header = ["Algorithm", "Score"]
        ##Print the table to the user
        print(tabulate(table, headers=header))

        """
        Time Complexity step count:
            2*i(Length of classifier list)
            loop occurs i times:
                1
                loop occurs k times:
                    3
                    3
                    4 + 2*n/k(Lists are of size n/k)
                7
                4
                2
                3
         Total Steps: 2*i+i+i*k*10+2in+16i
         Total Steps: 19i+10ik+2in
         
         BigO(n): c=i+3 n0=50
        """

    def confusionMatrix(self):
        """
        Creates a confusion matrix for each classifier

        Prints to console
        -------
        Creates confusion matrix comparing the test predicitions to the actual results.

        """
        ##For each classifier
        for i in range(len(self.ClassifierList)):
            ##Empty lists to store all predicted and true labels
            prediction = []
            true = []

            ##For each fold collect the labels and append to above lists
            for a in range(len(self.True_Labels[0])):
                predicted_labels = self.Predicted_Labels[i][a]
                true_labels = self.True_Labels[i][a].values
                prediction.append([max(c,key=c.get) for c in predicted_labels])
                true.append(true_labels)

            ##Flatten list of lists
            prediction = [z for sublist in prediction for z in sublist]
            true = [z for sublist in true for z in sublist]
            ##Count amount of different labels
            true_labels_count = len(np.unique(true))
            ##Collect unique labels
            labels = np.unique(true)
            ##Make numpy zeroes of dimensions # of labels*# of labels
            result = np.zeros((true_labels_count, true_labels_count))
            ##Create a dataframe of 0's with column and row indices as label names
            df = pd.DataFrame(result, index=labels, columns=labels)

            ##For each set of predicted and true labels
            for z in range(len(true)):
                ##Locate row of true label with column of predicted label and add 1
                df[true[z]][prediction[z]] += 1
            print(self.ClassifierList[i],"Confusion Matrix:",end="\n\n")
            print(df,end="\n\n")

        """
        Time Complexity Step Count:
            Loop occurs i times:
                2
                loop occurs k times:
                    3
                    3
                    1
                    1
                n
                n
                n+2
                n
                Number of unique labels(z)^2
                3
                loop occurs n times:
                    5
        Total step count: 2i+ ik(8) +i(4n+5+z^2) +5ni + i
        Total step count: 9i + 8ik + 4ni +iz^2
        
        BigO(n) c=i+5 n0=45
                
        """

    def ROC(self,positive_class_if_binary=None):
        '''
        

        Parameters
        ----------
        positive_class_if_binary : TYPE, optional
            DESCRIPTION. If dataset houses only 2 classes, enter the name of the 
            positive label in this format: [1] or ["string"]

        Returns
        -------
        ROC curve for all classifiers for all labels displayed in the console.

        '''
        ##For each classifier
        fig = plt.figure()
        ROC_Points = {}
        for i in range(len(self.ClassifierList)):
            ##Empty lists to store all predicted and true labels
            prediction = []
            true = []

            ##For each fold collect the labels and append to above lists
            for a in range(len(self.True_Labels[0])):
                predicted_labels = self.Predicted_Labels[i][a]
                true_labels = self.True_Labels[i][a].values
                prediction.append(predicted_labels)
                true.append(true_labels)

            ##Flatten list of lists
            prediction = [z for sublist in prediction for z in sublist]
            true = [z for sublist in true for z in sublist]
            number_of_classes = np.unique(true)
            if len(number_of_classes) == 2:
                ##Get label and confidence that it belongs to positive class
                prediction2=[(max(prop,key=prop.get),prop[positive_class_if_binary[0]]) for prop in prediction]
                ##Grab the confidence values
                values=[val[1] for val in prediction2]
                ##Sort in reverse order
                indices=list(np.argsort(values)[::-1])
                ##Sort predictions based on confidence
                prediction2=sorted(prediction2,key=lambda k:k[1],reverse=True)
                ##Sort true labels the same way the prediction labels were sorted
                true2=[true[indices[idx]] for idx in range(len(indices))]
                FP = 0
                TP = 0
                ##Counting how many positive and neagtive instances there are
                P = true2.count(positive_class_if_binary)
                nonpositive_class=[clss for clss in number_of_classes if clss not in positive_class_if_binary][0]
                N = true2.count(nonpositive_class)
                ##Empty list to store tuples of (FPR,TPR)
                R = []
                previous=-1
                ##For each label
                for z in range(len(true2)):
                    ##If likelihood doesn't equal last likelihood
                    if prediction2[z][1]!=previous or z==len(true2)-1:
                        ##Calculate FPR,TPR
                        FPR = FP / N
                        TPR = TP / P
                        R.append((FPR, TPR))
                        previous=prediction2[z][1]
                    ##If a true positive, increment
                    if true2[z] == positive_class_if_binary:
                        TP += 1
                    ##If a false positive, increment
                    else:
                        FP += 1
                ##Add points to dictionary with the classifier as the key
                ROC_Points[self.ClassifierList[i]] = R
                ##Plot points
                x = [point[0] for point in R]
                y = [point[1] for point in R]
                plt.plot(x, y, label=self.ClassifierList[i])

            else:
                ##Dictionary to store roc points for each class
                R_Multiple = {}
                for clss in number_of_classes:
                    ##Get tuple of predicted label and likelihood that row belongs to the class in question
                    prediction2=[(max(prop,key=prop.get),prop[clss]) for prop in prediction]
                    ##Sort predictions and true labels the same way -- based on confidences 
                    values=[val[1] for val in prediction2]
                    indices=list(np.argsort(values)[::-1])
                    prediction2=sorted(prediction2,key=lambda k:k[1],reverse=True)
                    true2=[true[indices[idx]] for idx in range(len(indices))]
                    FP = 0
                    TP = 0
                    P = true2.count(clss)
                    N = len(true2) - P
                    R = []
                    previous=-1
                    for z in range(len(true2)):
                        if prediction2[z][1]!=previous or z==len(true2)-1:
                            ##Calculate FPR,TPR
                            FPR = FP / N
                            TPR = TP / P
                            R.append((FPR, TPR))
                            previous=prediction2[z][1]
                        ##Increment if positive
                        if true2[z] == clss:
                            TP += 1
                        ##Increment if negative
                        else:
                            FP += 1
                    R_Multiple[clss] = R
                    x = [point[0] for point in R]
                    y = [point[1] for point in R]
                    ##Plot points for each class
                    plt.plot(x, y, label=str(self.ClassifierList[i] + " " + clss))
                ROC_Points[self.ClassifierList[i]] = R_Multiple
        ##Plot evaluation line
        plt.plot(
            [0, 1],
            [0, 1],
            label="Evaluation Line",
            color="black",
            linestyle="dashed",
        )
        plt.legend()
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.show()
        self.Points = ROC_Points

        
        '''
        Time Complexity Analysis:
        Line by Line Counts
        1
        2
        Loop that occurs c times (# of classifiers):
            2
            2
            Loop that occurs k times (# of folds in k-fold)
                3
                4
                1
                1
            n
            n
            n
            ---Worst case so looking at the condition where there are multiple classes----
            2
            Loop occurs t times (number of unique classes)
                1
                1
                5n
                2
                1
                loop occurs n times:
                    2
                    2
                    2
                    7
                2
                n
                n
                1
            2
        7
        
    Total Step Count: (3 + c*(4 + k*(9) + 3n + 2 + t*(5 + 5n + n*(13) + 3+ 2n) + 2)+1 + c*(1+t*n) + 7)
    Total Step Count : 11 + 9c + 9kc + 3nc + 8tc + 17ntc 
    Worst Case scenario : c = 5, k=10, t = 5
    --------------------------------------------------------------------
    Therefore Total Step Count: 706 + 566n  
    
    BigO(n) : c=567,  n0=706
    
    ---------------------------------------------------------------------
    Space Count Line By line Count:
        1
        c*t*n
        n
        n
        k
        k
        t
        t*n
        4
        n
        2
        n
        n
        1
        n
        1
        7
     ------------------------------------------------------------   
    Total Space Count: 16 + c*t*n + t*n +2k + t + 6n
    Worst Case scenario : c = 5, k=10, t = 5
    
    Thus Total Space Count: 31 + 36n 
    BigO(n): c=37 n0=31
    --------------------------------------------------------------------
            
        
        '''
