##Importing libraries
library(naivebayes)
library(ggplot2)
library(e1071)
library(dplyr)
library(caret)
library(cvms)
library(rsvg)
library(ggimage)
library(rpart)

##Importing data
DF<-read.csv('Clean_Data_UpdatedLabels.csv')
str(DF)
##Removing year and player name columns
DF<-DF[,-c(1,3)]
##Remove players who played less than 15 games 
DF<-filter(DF,DF$games_played>15)
##Changing variable types to factor
DF$SalaryLabel<-as.factor(DF$SalaryLabel)
DF$All_Nba<-as.factor(DF$All_Nba)
DF$All_Nba_team<-as.factor(DF$All_Nba_team)

##Assigning data to another dataframe which will be manipulated later
DF1<-DF
str(DF)
##Setting random seed for replicability when setting train and test sets
set.seed(4)
##Splitting into test and train
indexes<-sample(1:nrow(DF),as.integer(nrow(DF)/5))
TestDF<-DF[indexes,]
TrainDF<-DF[-indexes,]

##Plotting label counts to ensure balanced and representative data
ggplot(TrainDF,aes(x=SalaryLabel))+
  ggtitle('Train Set Labels')+
  geom_bar(fill='pink')
ggplot(TestDF,aes(x=SalaryLabel))+
  ggtitle('Test Set Labels')+
  geom_bar(fill='pink')
ggplot(DF,aes(x=SalaryLabel))+
  ggtitle('Data Set Labels')+
  geom_bar(fill='red')

##Isolating test labels
Testlabel<-TestDF$SalaryLabel
##Removing label and salary % columns
TrainDF<-TrainDF[,-which(names(TrainDF) %in% c('Salary_pct'))]
TestDF<-TestDF[,-which(names(TestDF) %in% c('SalaryLabel','Salary_pct'))]

##Running naive bayes model
(model<-naiveBayes(SalaryLabel~.,data=TrainDF,laplace = 1))

##Isolating team and position conditional probabilities
teamDF<-as.data.frame(model$tables$name)
positionDF<-as.data.frame(model$tables$primary_position)
##Reordering the labels
teamDF$Trainlabels<-factor(teamDF$Y,levels=c('below average','average','star','superstar'))
positionDF$Trainlabels<-factor(positionDF$Y,levels=c('below average','average','star','superstar'))

##Comparing conditional probabilites for Salary labels by team and position
ggplot(teamDF,aes(fill=Trainlabels,y=name,x=Freq))+
  geom_bar(position='fill',stat='identity')+
  ggtitle('Normalized Salary Label Conditional Probability By Team ')+
  xlab('Normalized Salary Label Conditional Probability')+
  ylab('Team Name')
##Position
ggplot(positionDF,aes(fill=Trainlabels,x=primary_position,y=Freq))+
  geom_bar(position='fill',stat='identity')+
  ggtitle('Normalized Salary Label Conditional Probability By Position ')+
  xlab('Position')+
  ylab('Normalized Salary Label Conditional Probability')




###Comparing accuracies for each model type using the best of each model


##Naive Bayes Feature Selected
DF1
str(DF1)
FeatureSelectedNB<-DF1[,-c(3,7,8,11,13:16)]
str(FeatureSelectedNB)

##Creating SVM ready data with min max norm and removing non-numeric columns
min_max_norm <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}
NormalizedDF<-as.data.frame(lapply(DF[,-c(1,2,18,19,24)],
                                   function(x) min_max_norm(x)))
##Adding factor columns back in
NormalizedDF$All_Nba<-DF$All_Nba
NormalizedDF$All_Nba_team<-DF$All_Nba_team
NormalizedDF$SalaryLabel<-DF$SalaryLabel

##Empty list to store model names and accuracies
model<-c()
accuracy<-c()

##Creating 20 different models each to plot accuracies
for (i in 1:20){
  ##Decision Tree
  
  ##Setting seed to create new model each time
  set.seed(i)
  ##Creating test and train set
  indexes<-sample(1:nrow(DF),as.integer(nrow(DF)/5))
  TestDF<-DF[indexes,]
  TrainDF<-DF[-indexes,]
  ##Isolating label
  TestlabelDT<-TestDF$SalaryLabel
  ##Removing name column and salary% column (as well as label in testDF)
  TrainDFDT<-TrainDF[,-which(names(TrainDF) %in% c('Salary_pct','name'))]
  TestDFDT<-TestDF[,-which(names(TestDF) %in% c('SalaryLabel','Salary_pct','name'))]
  
  ##Running best DT model and adding model name and accuracy to master lists
  DT<-rpart(SalaryLabel ~ .,data=TrainDFDT,cp=.02,method = 'class',
            parms=list(split='information'),minsplit=2)
  DTResults<-predict(DT,TestDFDT,type='class')
  DTcm<-confusionMatrix(DTResults,TestlabelDT)
  DTDF<-as.data.frame(DTcm$overall)
  model<-c(model,'DT')
  accuracy<-c(accuracy,DTDF[1,1])
  
  ##Naive Bayes
  set.seed(i)
  ##Train and test set partition
  indexesNB<-sample(1:nrow(FeatureSelectedNB),as.integer(nrow(FeatureSelectedNB)/5))
  TestDFNB<-FeatureSelectedNB[indexesNB,]
  TrainDFNB<-FeatureSelectedNB[-indexesNB,]
  TestlabelNB<-TestDFNB$SalaryLabel
  ##Removing salary%
  TrainDFNB<-TrainDFNB[,-which(names(TrainDFNB) %in% c('Salary_pct'))]
  TestDFNB<-TestDFNB[,-which(names(TestDFNB) %in% c('SalaryLabel','Salary_pct'))]
  ##Running NB feature selected
  NB<-naiveBayes(SalaryLabel~.,data=TrainDFNB,laplace = 1)
  NBResults<-predict(NB,TestDFNB)
  NBcm<-confusionMatrix(NBResults,TestlabelNB)
  NBDF<-as.data.frame(NBcm$overall)
  model<-c(model,'NB')
  accuracy<-c(accuracy,NBDF[1,1])
  
  ##Setting test and train sets
  set.seed(i)
  indexesSVM<-sample(1:nrow(NormalizedDF),as.integer(nrow(NormalizedDF)/5))
  TestDFSVM<-NormalizedDF[indexesSVM,]
  TrainDFSVM<-NormalizedDF[-indexesSVM,]
  ##isolating test label and removing salary % and label column 
  TestlabelSVM<-TestDFSVM$SalaryLabel
  TrainDFSVM<-TrainDFSVM[,-which(names(TrainDFSVM) %in% c('Salary_pct'))]
  TestDFSVM<-TestDFSVM[,-which(names(TestDFSVM) %in% c('SalaryLabel','Salary_pct'))]
  
  ##Running Radial SVM with cost of 10
  SVM<-svm(SalaryLabel~.,data=TrainDFSVM,kernel='radial',cost=10)
  SVMResults<-predict(SVM,TestDFSVM)
  SVMcm<-confusionMatrix(SVMResults,TestlabelSVM)
  SVMDF<-as.data.frame(SVMcm$overall)
  model<-c(model,'SVM')
  accuracy<-c(accuracy,SVMDF[1,1])
  
}
AccuracyDF<-data.frame(model,accuracy)
boxplot(accuracy~model,data=AccuracyDF,col=c('blue','red','green'),
        main='Model Accuracies',
        xlab='Model',
        ylab='Accuracy')
