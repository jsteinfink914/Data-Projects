##Importing Necessary Libraries
library(rpart) ##For decision trees
library(rattle)
library(RColorBrewer)
library(Cairo)
library(network)
library(ggplot2) ##For graphs
library(wordcloud)
library(tm)
library(slam)
library(quanteda)
library(proxy)
library(stringr)
library(textmineR)
library(igraph)
library(caret) ##For confusionMatrix
library(datasets)
library(dplyr)
library(maptree)
library(kableExtra)

##Reading in the data
DF<-read.csv('Clean_Data_UpdatedLabels.csv')


str(DF)
DF$SalaryLabel<-as.factor(DF$SalaryLabel)
##Removing name, team, year, and Salary percentage columns
DF<-DF[,-c(1:3,25)]
##Subsetting data by who has played over 15 games
DF<-subset(DF,games_played>=15)


##Plotting salary bins
ggplot(DF,aes(SalaryLabel))+
  ggtitle('Salary Bins')+
  geom_bar(fill='red')


##Splitting into test and train
datasize<-nrow(DF)
TrainingSet_Size<-floor(datasize*(3/4)) ##split of 75/25
TestSet_Size<-datasize-TrainingSet_Size
set.seed(15)
Trainsample<-sample(nrow(DF),TrainingSet_Size,replace=FALSE)
##Establishing the train dataset
Trainset<-DF[Trainsample,]

##Making sure train set is balanced
table(Trainset$SalaryLabel)
ggplot(Trainset,aes(SalaryLabel))+
  ggtitle('Train Set Labels')+
  geom_bar(fill='green')

##Create test set
Testset<-DF[-Trainsample,]
##Making sure test set is balanced
table(Testset$SalaryLabel)
ggplot(Testset,aes(SalaryLabel))+
  ggtitle('Test Set Labels')+
  geom_bar(fill='green')

##Remove the labels from test set
TestLabels<-Testset$SalaryLabel
Testset<-Testset[,-which(names(Testset) %in% c('SalaryLabel'))]

##Now onto Decision Trees
str(Trainset)
str(Testset)
##Using information gain and cp of.01
DT<-rpart(Trainset$SalaryLabel ~ .,data=Trainset,cp=.01,method = 'class',
           parms=list(split='information'),minsplit=2)
##Checking the results and cp
summary(DT)
plotcp(DT)

##Plotting decision Tree 1
fancyRpartPlot(DT,cex=.75,space=1,yspace=1,split.yshift=-1)

#Plotting Variable Importance
Variable<-as.data.frame(DT$variable.importance)
ggplot(Variable,aes(y=row.names(Variable),x=DT$variable.importance))+geom_bar(
  stat='identity',fill='orange')+
  xlab('Variables')+
  ylab('Importance')+
  ggtitle('Variable Importance')

##Using confusionMatrix to check accuracy

DT_prediction<-predict(DT,Testset,type='class')
caret::confusionMatrix(DT_prediction,TestLabels)
table(DT_prediction,TestLabels)  

##Second DT using gini splits and cp of .01
DT2<-rpart(Trainset$SalaryLabel ~ .,data=Trainset,cp=.01,method = 'class')
summary(DT2)
plotcp(DT2)

##Plotting decision Tree
fancyRpartPlot(DT2,cex=.75,space=1,yspace=1,split.yshift=-1)
#Plotting Variable Importance
Variable<-as.data.frame(DT2$variable.importance)
ggplot(Variable,aes(y=row.names(Variable),x=DT2$variable.importance))+geom_bar(
  stat='identity',fill='orange')+
  xlab('Variables')+
  ylab('Importance')+
  ggtitle('Variable Importance')

##Confusion Matrix for accuracy
DT2_prediction<-predict(DT2,Testset,type='class')
caret::confusionMatrix(DT2_prediction,TestLabels)
table(DT2_prediction,TestLabels)  



###3rd DT using information gain and cp of .02
DT3<-rpart(Trainset$SalaryLabel ~ .,data=Trainset,cp=.02,method = 'class',
           parms=list(split='information'),minsplit=2)
summary(DT3)
plotcp(DT3)
##Plotting decision Tree
fancyRpartPlot(DT3,cex=.75,space=1,yspace=1,split.yshift=-1)
#Plotting Variable Importance
Variable<-as.data.frame(DT3$variable.importance)
ggplot(Variable,aes(y=row.names(Variable),x=DT3$variable.importance))+geom_bar(
  stat='identity',fill='orange')+
  xlab('Variables')+
  ylab('Importance')+
  ggtitle('Variable Importance')

##Confusion Matrix for accuracy
DT3_prediction<-predict(DT3,Testset,type='class')
caret::confusionMatrix(DT3_prediction,TestLabels)
table(DT3_prediction,TestLabels)    
