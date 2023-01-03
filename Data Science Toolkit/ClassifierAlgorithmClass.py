# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 12:39:40 2022

@author: jstei
"""

from math import sqrt
import numpy as np


class ClassifierAlgorithm:
    def __init__(self):
        """

        Returns
        -------
        None.

        """

    def train(self, trainingData, Labels):
        """
        Function simply assigns trainingData and associated labels to
        object attributes.


        Parameters
        ----------
        trainingData : trainingData
        Labels: Labels for the trainingData

        Returns as attribute
        -------
        self.trainData: Training Data
        self.trainLabels: Training data lables


        """
        self.trainData, self.trainLabels = trainingData, Labels
        self.classes=np.unique(self.trainLabels)


class simpleKNNClassifier(ClassifierAlgorithm):
    def __init__(self):
        """
        Largely vacuous, simpliy initiates as a ClassifierAlgorithm object

        Returns
        -------
        None

        """
        super().__init__()

    def test(self, testData, testLabels, k=3):
        """
        Function takes training data and associated labels and predicts
        labels based on the training data. This function should be run after the
        train function.

        Parameters
        -----------
        k: Number of neightbords to look for when determining label


        Returns as attribute
        -------------------
        self.testData: testdata
        self.testLabels: Labels associated with the test data

        Returns
        -------
        self.PredictedLabels: dictionary of predicted labels and associated likelohoods
                                for the test data

        """
        self.testData, self.testLabels = testData, testLabels

        ##Empty list to store predicted labels
        Predicted_Labels = list()
        ##For each row in the test data...
        for i in range(len(self.testData)):
            Predicted_Labels.append(
                ##Append to the list the label computed in the __prediction function
                self.__prediction(self.trainData, self.testData.iloc[i].values, k)
            )
        ##Assign labels to object attribute and return
        self.Predicted_Labels = Predicted_Labels

        return self.Predicted_Labels

    def __euclidean_distance(self, row1, row2):
        """
        Private function used in tandem with __get_neighbors. Function is only used
        to calculate the euclidean distance between two rows of numeric data.

        Parameters
        ----------
        row1 : Row to be used to calucate distance
        row2 : Row to be used to calculate distance from row 1

        Returns
        -------
        distance: the euclidean distance between the two rows

        """
        distance = 0.0
        ##Loop throuhg every number in each row
        for i in range(len(row1) - 1):
            ##Calculate swuared distance and add to total
            distance += (row1[i] - row2[i]) ** 2
        ##Return euclidean distance
        return sqrt(distance)

    def __get_neighbors(self, train, test_row, k):
        """
        Function is used by __prediction to find the k nearest neighbors
        to the test row being examined


        Parameters
        ----------
        train : Train data
        test_row : Test set row of interest
        k: How mnay neighbors we want to find

        Returns
        -------
        the K nearest neighbors to the test row as a list of rows

        """
        ##List to store distances
        distances = list()
        ##Loop through every row in the train data
        for i in range(len(train)):
            ##Append the index of the row and its euclidean distance as tuple
            ##to the empty lsit
            dist = self.__euclidean_distance(test_row, train.iloc[i].values)
            distances.append((i, dist))

        ##Sort the list based on distance in descending order
        distances.sort(key=lambda tup: tup[1])

        ##List to store closest neighboring rows
        neighbors = list()
        ##Append indexes of nearest rows
        for i in range(k):
            neighbors.append(distances[i][0])
        return neighbors
    
    def __class_makeup(self,labels,count):
        
        class_percents={}
        for i in range(len(labels)):
            class_percents[labels[i]] = count[i]/sum(count) 
        if len(labels)!= len(self.classes):
            new_labels=[c for c in self.classes if c not in labels]
            for lab in new_labels:
                class_percents[lab]=0
        
        return class_percents
         
        
            
        
        

    def __prediction(self, train, test_row, k):
        """
        Makes use of __euclidean_distance and __get_neighbors functions to
        compute the mode label of the nearest neighbors

        Parameters
        ----------
        train : Train data
        test_row : Test set row of interest
        k: How mnay neighbors we want to find

        Returns
        -------
        prediction: a dictionary of labels and the confidence that the row belongs
                    to that label compiled from the k nearest neighbors

        """
        ##Find the k nearest neighbors to the test row
        neighbors = self.__get_neighbors(train, test_row, k)
        ##Gather the labels and the counts of the labels
        labels, count = np.unique(self.trainLabels.iloc[neighbors], return_counts=True)
        ##Get the label proportions,returned as a set
        class_percents = self.__class_makeup(labels,count)
        prediction = class_percents
        return prediction

    """
    Time Complexity for simpleKNNCLassifier Test Method Line by line counts:
    2
    1
    ##Loop through test data:
    Length of test data(z)*(2)
        ##Function call to neighbors happens m times
            1
            length of train(i):
                3
                ##Function call to euclidean distance happens i times
                    2+size of row(m)*3
                    
        2i+2
        2
        2
        1
    2
    Total time steps: 2+1+2z+z(3i)+z*i*(2+3m)+z(7+2i)+2
    Total time steps: 5+9z+7zm+3izm where z=test set length,i=train set length,m=length of rows
    
    Total data is (i+z)*m dimensions, i+z=n
    
    T(n): 5+9z+7zi+3izm
    
    Big O(m*n) c=0 n0=1
                    
            
    
    """


class Tree:
    '''
    This class is used to maintain the decsion tree built in the DecisionTreeClassifier
    Class. Nodes hold indices as keys, labels associated with those indices as values,
    a dictionary holding the decision present at that node, as well as left and right 
    child nodes, and parent node.
    '''
    def __init__(self, key, val, decision=None, left=None, right=None, parent=None):
        self.key = key
        self.payload = val
        self.decision = decision
        self.leftChild = left
        self.rightChild = right
        self.parent = parent

    def hasLeftChild(self):
        return self.leftChild

    def hasRightChild(self):
        return self.rightChild

    def isLeftChild(self):
        return self.parent and self.parent.leftChild == self

    def isRightChild(self):
        return self.parent and self.parent.rightChild == self

    def isRoot(self):
        return not self.parent

    def isLeaf(self):
        return not (self.rightChild or self.leftChild)

    def hasAnyChildren(self):
        return self.rightChild or self.leftChild

    def hasBothChildren(self):
        return self.rightChild and self.leftChild

    def spliceOut(self):
        if self.isLeaf():
            if self.isLeftChild():
                self.parent.leftChild = None
            else:
                self.parent.rightChild = None
        elif self.hasAnyChildren():
            if self.hasLeftChild():
                if self.isLeftChild():
                    self.parent.leftChild = self.leftChild
                else:
                    self.parent.rightChild = self.leftChild
                self.leftChild.parent = self.parent
            else:
                if self.isLeftChild():
                    self.parent.leftChild = self.rightChild
                else:
                    self.parent.rightChild = self.rightChild
                self.rightChild.parent = self.parent

    def findSuccessor(self):
        succ = None
        if self.hasRightChild():
            succ = self.rightChild.findMin()
        else:
            if self.parent:
                if self.isLeftChild():
                    succ = self.parent
                else:
                    self.parent.rightChild = None
                    succ = self.parent.findSuccessor()
                    self.parent.rightChild = self
        return succ

    def findMin(self):
        current = self
        while current.hasLeftChild():
            current = current.leftChild
        return current

    def replaceNodeData(self, key, value, lc, rc):
        self.key = key
        self.payload = value
        self.leftChild = lc
        self.rightChild = rc
        if self.hasLeftChild():
            self.leftChild.parent = self
        if self.hasRightChild():
            self.rightChild.parent = self

    def __str__(self):
        return self.toString(self.root)

    # Recursive Depth First Traversal that includes string viz to be used
    # with website:  http://mshang.ca/syntree/
    def toString(self, thisNode):
        if thisNode == None:
            return "[None]"
        lhs = self.toString(thisNode.leftChild)
        rhs = self.toString(thisNode.rightChild)
        return "[" + str({k: thisNode.decision[k] for k in thisNode.decision.keys() - {'groups'}}) + str(lhs) + str(rhs) + "]"


class DecisionTreeClassifier(Tree,ClassifierAlgorithm):
    def __init__(self):
        """

        Returns
        --------
        None.

        """
        super(ClassifierAlgorithm).__init__()

    def train(self, traindata, trainlabels, max_depth=5, min_size=10):
        """
        Creates a decision tree using gini index as criteria for splitting.
        Nodes of the tree will store the indexes of the data they house as the key,
        and the associated training labels as the payload. A new attribute called decision
        will either store a dictionary of the split that occurs at the node, or,
        if a terminal node, it will return the label that is most represented.

        Parameters
        ----------
        traindata : Unlabeled Training data for the algorithm
        trainlabels : Labels belonging to the train data
        max_depth : TYPE, optional depth of the tree. The default is 5.
        min_size : TYPE, optional; minimum number of points required for a split. The default is 10.

        Returns
        -------
        Printed version of the decision tree to the console.

        Attributes Created
        ------------------
        self.train: Train data
        self.trainLabels: Train Labels
        self.T: Decision Tree
        """
        self.train = traindata
        self.trainLabels = trainlabels
        self.classes=np.unique(self.trainLabels)
        ##Initializes root node of tree
        self.T = Tree(list(self.train.index), self.trainLabels)
        self.T.root = self.T
        ##Makes a function call to build_tree()
        self.build_tree(self.T, max_depth, min_size)
        # print(self.T)
        '''
        Line by Line Step Count:
        1
        1
        6
        1
        1
        Enter Build tree function:
        1
        Enter Get Split function:
        4
        Loop occurs m times
            Loop occurs logn times
                1
                Enter Test Split function
                2
                Loop occurs logn times
                    2
                    1
                Exit Test Split
                Enter gini index function
                n
                1
                Loop occurs 2 times
                    3 
                    Loop occurs t times
                        log n 
                        log n
                        2
                    5
                Exit gini index
                6
                Exit get split
        Enter split function
        Occurs 2^max_depth times:
            6
            1
            1
            4
            6
            1
            Perform Get Split
            Perform Split 
        last round of split function
        5
        Enter to terminal function
        3 log n
        
        Total Step Count: 11 + get_split step count + split function + ,5 + 3 log n
        Get_split step count: 4 + m*logn(3 + log n*(3) + n + 1 + 2*(8 +t*(2log n + 2)) + 6)
        Split step count: 2^max_depth * (17 + get_split)
        -----------------------------------------------------------------------------------------------
        Total Step Count: 11 + 4 + 3mlogn + 3mlog^2 n + m(logn)^2 + mlogn + 16mlogn + 4mt(logn)^2 + 4mlognt  + 6mlogn + 2^max_depth*17 + 2^max_depth * (4 + 3mlogn + 3m(log n)^2 +mnlogn + mlogn + 16mlogn + 4tm(log n)^2 + 4tmlogn + 6mlogn)
        T(n): 15 + 27mlogn + 3m(log n)^2  + 4mt(log n)^2 + 4mtlogn + 2^max_depth * 17  + 2^max_depth * (4 + 3mlogn + 3m(log n)^2 +mnlogn + mlogn + 16mlogn + 4tm(log n)^2 + 4tmlogn + 6mlogn)
        
        Final Step Count T(n): 687 + 1518mlogn + 755m(log n)^2 + 33mnlogn 
        
        Big O(n^2): c=m n0=200
        
        --------------------------------------------------------------------------------------
        
        
        
        Space Count:
        1
        1
        6
        Enter Get_split
        4
        1
        Enter test split
        log n
        1
        Exit test split
        Enter gini
        log n
        1
        1
        1
        1
        1
        1
        Exit gini
        4
        Exit get_split
        Enter split - occurs 2^max_depth times
        log n
        4
        4
        6
        1
        Run get_split
        Run Split again
        Last occurence of split, happens 2^ max_depth times
        log n
        4
        4
        
        ------------------------------------------------------------------------
        Space Count S(n): 24 + 2log n + 2^max_depth*(4log n + 39)
         Worst Case scenario : t = 5, max_depth=5
         S(n): 1272 + 130log n 
         Big O(log n): c=550 n0= 1000
        ---------------------------------------------------------------------------
        
        '''

    def test(self, testdata):
        """

        Parameters
        ----------
        testdata : Unlabeled data seeking prediction labels

        Returns
        -------
        self.Predicted_Labels: Predicted labels created by running through decision tree;
                                returns a dictionary with each class and the confidence that the item
                                belongs to that class

        """
        ##Resetting the index
        self.test = testdata.reset_index(drop=True)
        ##Empty list to store predictions
        predictions = list()
        ##For each row in the test data
        for i in range(len(self.test)):
            ##Predict what label it should have and add to list
            prediction = self.__predict(self.T, self.test.loc[i])
            predictions.append(prediction)
        ##Create and return the Predicted_Labels Attribute
        self.Predicted_Labels = predictions
        return self.Predicted_Labels
    
        '''
        Line by line Step Count:
        1
        1
        loop occurs n*.2 times:
            2
            Enter predict function
            5
            1
            recur potentially max_depth times
            2
            Exit predict function
            1
        1
        1
        --------------------------------------------------------------
        Step Count: 1+ n*.2(2 + 6*max_depth + 2 + 1) + 2
        Worst case: Max_depth= 5
        T(n): 3 + 7n
        Big O(n): c=8 n0=3
        
        -----------------------------------------------------------------
        
        Space Count:
        n*.2 *m
        n*.2
        Enter predict:
        2
        ---------------------------------------------------------
        Space Count: m*n*.2 + .4n +2
        BigO(m*n): c=1 n0=1
        ----------------------------------------------------
        '''
    

    def __test_split(self, index, value, rows):
        ##Lists that will hold indices of each group
        left, right = list(), list()
        ##For each index
        for row in rows:
            ##If the row in question is less than the potential split value
            if self.train.loc[row][index] < value:
                ##Assign it to the left node
                left.append(row)
            else:
                ##Else assign to right group
                right.append(row)
        return left, right

    # Calculate the Gini index for a split dataset
    def __gini_index(self, node, group1, group2):
        ##Get number of instances at the node
        n_instances = float(len(group1) + len(group2))
        ##Will hold the weighted gini index (weighted by size of each group)
        gini = 0.0
        ##For each group --left and right-- in the potential split
        for group in [group1, group2]:
            ##Get size of the group
            size = float(len(group))
            ##Avoid dividing by zero and empty split
            if size == 0:
                continue
            ##Set the score which will hold each classes gini index
            score = 0.0
            ##For each class in the node
            for class_val in np.unique(node.payload):
                ##Get array of the labels represented in the whole node
                arr = node.payload.loc[group].values
                ##Count $ of times the label is represented in the subgroup
                p = np.count_nonzero(arr == class_val) / size
                ##Square this number and add to score
                score += p * p
            ###Added weighted gini index for each class to total score
            gini += (1.0 - score) * (size / n_instances)
        return gini

    # Select the best split point for a dataset
    def __get_split(self, node):
        ##Initializing attributes which will eventually store the best split
        b_column, b_value, b_score, b_groups = None, None, 2, None
        ##For each column in the dataframe
        for column_index in range(len(self.train.columns.tolist())):
            ##For each row index in the node
            for row in node.key:
                ##Get a row of data in the node
                data = self.train.loc[row]
                ##Use the __test_split() function to obtain potential left and right groups
                ##Passing the column index and the potential value of the split and
                ##all of the indexes in the node
                group1, group2 = self.__test_split(
                    column_index, data[column_index], node.key
                )
                ##Calculate the gini index of the potential split
                gini = self.__gini_index(node, group1, group2)
                if gini < b_score:
                    b_column, b_value, b_score, b_groups = (
                        column_index,
                        data[column_index],
                        gini,
                        [group1, group2],
                    )
        return {"index": b_column, "value": b_value, "groups": b_groups, "gini": gini}

    # Create child splits for a node or make terminal
    def __split(self, node, best_split, depth, max_depth=5, min_size=10):
        left, right = best_split["groups"][0], best_split["groups"][1]
        ##Store the split as a decision
        node.decision = best_split
        ##If the node is pure...
        if node.decision["gini"] == 0:
            ##Don't split further and store label as decision
            node.decision = self.__to_terminal(node)
            return
        ##check for a no split
        if len(left) == 0 or len(right) == 0:
            return
        ##check for max depth
        if depth >= max_depth:
            node.decision = self.__to_terminal(node)
            return
        ##If size of the node is too small for a split
        if len(left) <= min_size:
            ##Create child node
            nodeL = Tree(left, self.trainLabels.loc[left], decision=None, parent=node)
            ##Make its decision the predominant label
            nodeL.decision = self.__to_terminal(nodeL)
            ##Assign as left child node
            node.leftChild = nodeL
        else:
            ##Crate left child node
            nodeL = Tree(left, self.trainLabels.loc[left], decision=None, parent=node)
            node.leftChild = nodeL
            ##Get its next split and recur
            next_split = self.__get_split(node.leftChild)
            self.__split(node.leftChild, next_split, depth + 1, max_depth, min_size)
        ##If node is too small to split
        if len(right) <= min_size:
            ##Create child node
            nodeR = Tree(right, self.trainLabels.loc[right], decision=None, parent=node)
            ##Get predominant label and set as decision
            nodeR.decision = self.__to_terminal(nodeR)
            node.rightChild = nodeR
        else:
            ##Create new node, assign as right child and recur
            nodeR = Tree(right, self.trainLabels.loc[right], decision=None, parent=node)
            node.rightChild = nodeR
            next_split = self.__get_split(node.rightChild)
            self.__split(node.rightChild, next_split, depth + 1, max_depth, min_size)

    def __to_terminal(self, node):
        ##Return a set of class percentages
        class_percents={}
        labels,count = np.unique(node.payload, return_counts=True)
        for i in range(len(labels)):
            class_percents[labels[i]] = count[i]/sum(count)
        if len(labels)!= len(self.classes):
            new_labels=[c for c in self.classes if c not in labels]
            for lab in new_labels:
                class_percents[lab]=0
        
        return class_percents

    # Build a decision tree
    def build_tree(self, node, max_depth, min_size):
        ##Calls the __get_split() method to obtain the optimal split for the node
        best_split = self.__get_split(node)
        self.best_split = best_split
        ##Calls the __split() method which recursively performs splits to build the tree
        self.__split(node, best_split, 1, max_depth, min_size)

    def __predict(self, tree, row):
        ##If the rows column index is less then the decision value
        if row[tree.decision["index"]] < tree.decision["value"]:
            ##If the left child is not a terminal node
            if isinstance(tree.leftChild.decision, dict) and "index" in tree.leftChild.decision.keys() :
                ##recur
                return self.__predict(tree.leftChild, row)
            else:
                ##Take the label
                return tree.leftChild.decision
        else:
            ##If the rows column index is greater than decision value
            ##And right child is not a terminal node
            if isinstance(tree.rightChild.decision, dict) and "index" in tree.rightChild.decision.keys():
                ##Recur
                return self.__predict(tree.rightChild, row)
            ##If right child is terminal, take its label
            else:
                return tree.rightChild.decision
