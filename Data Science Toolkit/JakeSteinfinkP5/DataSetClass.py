# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 17:50:01 2022

@author: jstei
"""

# * JakeSteinfinkP1.py
# *
# *  ANLY 555 2022
# *  Project Deliverable 2
# *
# *  Due on: 2/27/2022
# *  Author(s): Jake Steinfink
# *
# *
# *  In accordance with the class policies and Georgetown's
# *  Honor Code, I certify that, with the exception of the
# *  class resources and those items noted below, I have neither
# *  given nor received any assistance on this project other than
# *  the TAs, professor, textbook and teammates.
# *
##
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from wordcloud import WordCloud


class DataSet:
    def __init__(self, filename, filetype="csv"):
        """
        Makes use of the __readFromCSV(self,filename) or __load(self,filename)
        methods to instantiate the data object

        Parameters
        ----------
        filename : file to be used

        filetype: default is csv-- file type can be any value. Anything but csv
                    will be loaded using the .read() function

        Returns
        -------
        self.data: the data object is stored in the .data attribute

        """
        if filetype == "csv":
            self.data = self.__readFromCSV(filename)
        else:
            self.data = self.__load(filename)

    def __readFromCSV(self, filename):
        """

        Parameters
        ----------
        filename : csv file to be read using pandas library - pd.read_csv()

        Returns
        -------
        .data attribute: the information from the csv file as a pandas dataframe.

        """
        x = pd.read_csv(filename)
        return x

    def __load(self, filename):
        """

        Parameters
        ----------
        filename :  file to be loaded using the .read() function.
                    This is meant for text files

        Returns
        -------
        .data attribute which will store a list of list of words

        """
        with open(filename, "r") as file:
            contents = file.read().replace("\n", "")

        return contents

    def findMedian(self, values):
        """


        Parameters
        ----------
        values : List of values where the median must be found

        Returns
        -------
        median : Median number in the list

        """
        ##Sorting values in ascending order
        for i in range(len(values) - 1):
            a = values[i]
            b = i
            for j in range(i + 1, len(values)):
                if values[j] < a:
                    a = values[j]
                    b = j
            values[i], values[b] = values[b], values[i]
        ##If list is of even length
        if len(values) % 2 == 0:
            ##Store index of halfway point
            index = len(values) // 2
            ##Collect 2 values at the midpoint
            number1 = values[index]
            number2 = values[index - 1]
            ##Average of values = median
            median = (number1 + number2) / 2
        else:
            ##If the list is of odd length just take the middle #
            median = values[len(values) // 2]
            ##Return the median
        return median


class TimeSeriesDataSet(DataSet):
    def __init__(self, filename):
        """

        Parameters
        ----------
        filename : file to be used

        Makes use of the __readFromCSV(self,filename)
        method to instantiate the data object from the SuperClass DataSet()

        Returns
        -------
        self.data: the data object is stored in the .data attribute

        """
        super().__init__(filename)

    def clean(self, filter_size=3, Columns=""):
        """
        TimeSeriesDataSet.clean() applies a median filter to the data to reduce
        noise. The function iteratively creates a list of the (filter_size-1)/2
        preceding and following points, takes the median and returns that as the value


        Parameters
        ---------
        filter_size: Determines size of the median filter, must be odd;
        default size is 3

        Columns: Columns for the filter to be applied to; default is all

        Creates
        -------
        Edits .data to be a clean (noise reduced) version of the dataframe.

        """
        ##Checking columns and setting to all columns if nothing provided
        if Columns == "":
            Columns = self.data.columns.tolist()
        ##Checking if median filter is odd
        if filter_size == 1:
            pass
        elif filter_size % 2 != 0:
            ##Empty list for values to be appended to
            temp = []
            ##Will be used to collect preceding and subsequent data points
            indexer = filter_size // 2

            ##Looping through each column and then each row
            for i in Columns:
                ##Will hold the changed data for each column
                data_final = []
                for j in range(len(self.data)):
                    ##If at the top of the dataframe
                    if j - indexer < 0:
                        ##Add 0s for how many points are needed to fill
                        ##the filter
                        for c in range(abs(j - indexer)):
                            temp.append(0)
                        ##Fill in the rest with appropriate values
                        for d in range(indexer + 1):
                            ##For filling in the actual point of interest
                            if d == 0:
                                temp.append(self.data[i].loc[j])
                            ##If this value in the data and precedes the
                            ##point of interest add the value
                            else:
                                if j - d >= 0:
                                    temp.append(self.data[i].loc[int(j - d)])
                                ##If this value in the data and is after the
                                ##point of interest add the value
                                if j + d <= len(self.data) - 1:
                                    temp.append(self.data[i][int(j + d)])
                                ##Special case if the point's filter spans both
                                ##the beginning and end of the dataframe
                                if j + d > len(self.data) - 1:
                                    temp.append(0)
                    ##If at the bottom of the dataframe
                    elif j + indexer > len(self.data) - 1:
                        ##Add 0s for how many points are needed to fill
                        ##the filter
                        for c in range(j + indexer - len(self.data) + 1):
                            temp.append(0)
                        ##Fill in the rest with appropriate values
                        for d in range(indexer + 1):
                            ##For filling in the actual point of interest
                            if d == 0:
                                temp.append(self.data[i].loc[j])
                            else:
                                ##If this value in the data and precedes the
                                ##point of interest add the value
                                if j - d >= 0:
                                    temp.append(self.data[i][int(j - d)])
                                ##If this value in the data and is after the
                                ##point of interest add the value
                                if j + d <= len(self.data) - 1:
                                    temp.append(self.data[i][int(j + d)])
                    ##If an entirely interior point given the filter
                    else:
                        ##Add all the points in the filter to the temp list
                        for d in range(indexer + 1):
                            if d == 0:
                                temp.append(self.data[i][j])
                            else:
                                temp.append(self.data[i][int(j - d)])
                                temp.append(self.data[i][int(j + d)])

                    ##Sort surounding median filter points in ascending order
                    temp.sort()
                    ##Append to the data the median point in the filter
                    data_final.append(temp[len(temp) // 2])
                    ##Reset temp
                    temp = []
                self.data[i].replace(self.data[i].values, data_final, inplace=True)
        else:
            raise TypeError("Filter Size cannot be an odd number.")

    def explore(self, Columns):
        """
        Parameters
        --------------
        Columns: list of two column names [x,y] to visualize; x should be a
                  time variable that can be converted to a datetime object;
                  y should be the numeric variable of interest


        Returns + Displays
        -------
        Viz1:  line chart of the y variable over time
        Viz2:  line chart of the Rate of change of the y variable over time


        """
        ##################################################################
        ##VIZ 1
        ################################################################
        ##Making line chart of x vs. y variable

        ##Making time variable a datetimes object
        self.data[Columns[0]] = pd.to_datetime(self.data[Columns[0]])
        ##Sort variable by time in ascending order
        self.data = self.data.sort_values(by=Columns[0])
        ##Collect values and plot
        x = self.data[Columns[0]].tolist()
        y = self.data[Columns[1]].tolist()
        xaxis_label = "Time" + " " + Columns[0]
        yaxis_label = Columns[1]
        title = Columns[1] + " over " + Columns[0]
        fig1 = plt.figure()
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel(xaxis_label)
        plt.ylabel(yaxis_label)
        plt.show()

        ##################################################################
        ##VIZ 2
        ################################################################

        ##Making line chart of rate of change of x vs. y

        ##Removing first value because it will be NaN
        x = self.data[Columns[1]].pct_change().tolist()[1:]
        y = self.data[Columns[0]].tolist()[1:]
        xaxis_label = "Time" + " " + Columns[0]
        yaxis_label = "Rate of Change of" + " " + Columns[1]
        title = "Rate of Change of" + " " + Columns[1] + " " + "Over Time"
        fig2 = plt.figure()
        plt.plot(y, x)
        plt.xlabel(xaxis_label)
        plt.ylabel(yaxis_label)
        plt.title(title)
        plt.show()

        return fig1, fig2


class TextDataSet(DataSet):
    def __init__(self, filename, filetype):
        """

        Parameters
        ----------
        filename : file to be used

        Makes use of the __readFromCSV(self,filename) or __load(self,filename)
        methods to instantiate the data object

        filetype: Paramater meant specifically for text analysis, filetype should
                    be anything other than 'csv'

        Returns
        -------
        self.data: A list of lists of words stored in the .data attribute

        """
        super().__init__(filename, filetype)

    def clean(self, Stopwords="", language="english"):
        """
        Parameters
        ----------
        Stopwords: a list of stop words to be removed from the text file.
                    Will be added on top stop words from nltk stopwords
        language: language the document is in; default is english

        Creates
        -------
        self.words: the words in the document without stopwords as a list.
                    This is not returned to the user but rather creates this attribute


        """
        ##Setting stop words
        stop_words = list(set(stopwords.words(language)))
        ##Adding user provided stopwords to set, if they exist
        if Stopwords != "":
            for i in Stopwords:
                stop_words.append(i)
            stop_words = list(set(stop_words))
        ##Tokenizing the words in the data
        ##Converting all words to one list, all lower case
        ##Tokenizing
        tokenizer = RegexpTokenizer(r"\w+")
        clean_words = tokenizer.tokenize(self.data)
        # tokens=word_tokenize(clean_words)
        ##Converting everything to lower case
        ##Filtering out stop words
        filtered_words = [i for i in clean_words if i not in stop_words]
        filtered_words = [i for i in filtered_words if len(i) > 3 & len(i) < 11]
        ##Returning cleaned words under the .words attribute
        self.words = filtered_words

    def explore(self,Columns=''):
        """

        Returns + Displays
        --------------------
        Viz1: Wordcloud of the data
        Viz2: Top 20 words by frequency count

        """
        ##################################################################
        ##VIZ 1
        ################################################################
        text = " ".join(self.words)
        word_cloud = WordCloud(collocations=False).generate(text)
        plt.imshow(word_cloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

        ##################################################################
        ##VIZ 2
        ################################################################

        ##Creating an ordered dictionary in descending order of the word frequencies
        top_words_dict = {
            i: j
            for i, j in sorted(
                word_cloud.words_.items(), key=lambda x: x[1], reverse=True
            )
        }
        ##Choose top 20 words and their frequencies
        top20_words = list(top_words_dict.keys())[:19]
        top20_words_frequencies = list(top_words_dict.values())[:19]

        ##Plotting the figure
        fig2 = plt.figure()
        plt.barh(top20_words, top20_words_frequencies)
        plt.xlabel("Word Frequencies")
        plt.ylabel("Top Words")
        plt.title("Top 20 Word Frequencies")
        plt.show()

        return word_cloud, fig2


class QuantDataSet(DataSet):
    def __init__(self, filename):
        """

        Parameters
        ----------
        filename : file to be used

        Makes use of the __readFromCSV(self,filename)
        method to instantiate the data object

        Returns
        -------
        self.data: the data object is stored in the .data attribute.

        """
        super().__init__(filename)

    def clean(self, Columns=""):
        """
        Parameters
        ----------
        Columns: a list of columns for the function to be applied to ;
                 default is all the numeric columns in the data

        Creates
        -------
        self.data: Returns a version of the dataframe with missing
        values replaced with the mean

        """
        ##Checking columns and setting to all columns if nothing provided
        if Columns == "":
            Columns = self.data.select_dtypes(include="number").columns.tolist()
        ##Looping through each column
        for i in Columns:
            ##Calculating the mean
            mean_value = self.data[i].sum() / len(self.data[i])
            ##Replacing missing values with the mean
            self.data[i].fillna(value=mean_value, inplace=True)
        ##Returning the data object with missing values filled

    def explore(self, Columns):
        """
        Parameters
        ----------
        Columns: 2 columns in [x,y] form to be plotted

        Returns + Displays
        --------------------
        Viz1: Scatterplot of x vs. y
        Viz2: Bar chart of x vs. y

        """
        ##################################################################
        ##VIZ 1
        ################################################################
        x = self.data[Columns[0]].tolist()
        y = self.data[Columns[1]].tolist()
        xlabel = Columns[0]
        ylabel = Columns[1]
        title1 = "Scatter Plot of" + " " + xlabel + " and " + ylabel
        fig1 = plt.figure()
        plt.scatter(x, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title1)
        plt.show()

        ##################################################################
        ##VIZ 2
        ################################################################
        title2 = "Bar Plot of" + " " + xlabel + " and " + ylabel
        fig2 = plt.figure()
        plt.bar(x, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title2)
        plt.show()

        return fig1, fig2


class QualDataSet(DataSet):
    def __init__(self, filename):
        """

        Parameters
        ----------
        filename : file to be used

        Makes use of the __readFromCSV(self,filename)
        method to instantiate the data object

        Returns
        -------
        self.data: The data object is stored in the .data attribute.

        """
        super().__init__(filename)

    def clean(self, Columns="", value="mode"):
        """
         Paramters
         -----------
         Columns: a list of columns for the function to be applied to;
                  default is all columns

         value: can equal "median" or "mode"; default is 'mode'

         Creates
         -------
        self.data: a version of the dataframe with missing
        values replaced with the median or mode
        """
        ##Checking columns and setting to all columns if nothing provided
        if Columns == "":
            Columns = self.data.columns.tolist()

        ##If mode is the choice
        if value == "mode":
            ##Loop through columns
            for i in Columns:
                mode_value = self.data[i].mode()[0]
                ##Filling missing values with the mode
                self.data[i].fillna(value=mode_value, inplace=True)
        else:
            ##Fill mssing vals with the median, for all chosen columns
            for i in Columns:
                ##Taking all numeric values from each column
                numeric_values = (
                    self.data[i]
                    .astype("str")
                    .str.extract("(\d*\.\d+|\d+)", expand=False)
                    .astype(float)
                )
                ##Making a function call to findMedian function in the SuperClass
                median_value = self.findMedian(numeric_values.tolist())
                self.data[i].fillna(value=median_value, inplace=True)

    def explore(self, Columns=""):
        """
        Parameters
        ------------
        Columns: Columns to be vizualized; default set to all. If default is used,
                    only second plot is displayed due to memory overloading
                    for high dimensional data

        Displays
        -------
        Viz1: Histogram for each column of interest
        Viz2: Graph of unique value counts for each column

        Returns
        ---------
        fig2: A barplot of the count of unique values for each column

        """
        if Columns == "":
            Columns = self.data.columns.tolist()
        ##################################################################
        ##VIZ 1
        ################################################################
        else:
            for i in Columns:
                plt.subplots(1, 1, figsize=(15, 6))
                x = self.data[i].tolist()
                title = "Histogram of " + i
                plt.hist(x)
                plt.xlabel(i)
                plt.ylabel("Count")
                plt.title(title)
            plt.show()
        ##################################################################
        ##VIZ 2
        ################################################################
        counts = []
        for i in Columns:
            x = self.data[i].tolist()
            unique_values = list(set(x))
            count = len(unique_values)
            counts.append(count)
        fig2 = plt.figure()
        plt.barh(Columns, counts)
        plt.ylabel("Columns")
        plt.xlabel("Unique Values Count")
        plt.title("Unique Values Count for each Column")
        plt.show()
        return fig2

class HeterogeneousData(DataSet):
    
    def __init__(self,df_list):
        '''
        Parameters
        ----------
        df_list : Dictionary of already instatiated class object to be used with
                    the objects as keys and the columns of interest for viz as values


        Returns as attribute
        -------
        self.data_list: List of class objects that can be used for further exploration
        
        '''
        self.data_list=df_list
        
    def clean_all(self):
        '''
        Runs the .clean() method for each class object in the list

        '''
        for df in self.data_list.keys():
            df.clean()
            
    def explore_all(self):
        '''
        Runs the .explore() method for each class object in the list

        '''
        for df in self.data_list.keys():
            df.explore(self.data_list[df])
            
    def select(self,index):
        '''
        

        Parameters
        ----------
        index : Index of the class object you want to investigate 

        Returns
        -------
        The class object at the index=index position in the .data_list

        '''
        self.select=list(self.data_list.keys())[index]
        return self.select
        
        
            
        
