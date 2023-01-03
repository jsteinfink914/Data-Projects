##This code runs 2 naive bayes models on mixed record data,
##one with feature selection and one without.
##Labels are plotted to ensure balanced 
##train and test sets.

##Importing libraries
library(naivebayes)
library(ggplot2)
library(e1071)
library(dplyr)
library(caret)
library(cvms)
library(rsvg)
library(ggimage)


##Importing data
DF<-read.csv('Clean_Data_UpdatedLabels.csv')
str(DF)
##Removing year and player name columns
DF<-DF[,-c(1,3)]
##Remove players who played less than 15 games 
DF<-filter(DF,DF$games_played>15)
DF$SalaryLabel<-as.factor(DF$SalaryLabel)
DF$All_Nba<-as.factor(DF$All_Nba)
DF$All_Nba_team<-as.factor(DF$All_Nba_team)
DF1<-DF
str(DF)
##Setting random seed for replicability when setting train and test sets
set.seed(4)
indexes<-sample(1:nrow(DF),as.integer(nrow(DF)/5),replace=FALSE)
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

##Isolating train and test labels
Trainlabels<-TrainDF$SalaryLabel
Testlabel<-TestDF$SalaryLabel
TrainDF<-TrainDF[,-which(names(TrainDF) %in% c('SalaryLabel','Salary_pct'))]
TestDF<-TestDF[,-which(names(TestDF) %in% c('SalaryLabel','Salary_pct'))]

##Running naive bayes model
(model<-naiveBayes(TrainDF,Trainlabels,laplace = 1))
pred<-predict(model,TestDF)
##Creating confusion matrix
cm<-confusionMatrix(pred,Testlabel)
cmDF<-as.data.frame(cm$table)
plot_confusion_matrix(cmDF, 
                      target_col = "Reference", 
                      prediction_col = "Prediction",
                      counts_col = "Freq",
                      add_row_percentages = FALSE,
                      add_col_percentages = FALSE,
                      rm_zero_percentages = FALSE,
                      rm_zero_text = FALSE,
                      add_zero_shading = TRUE,
                      counts_on_top = TRUE,
                      palette='Blues') 

##Drop games_played,assists_turnover,points_off_turnovers,steals
##height,weight, effective_fg_pct, true shooting %

##Process is repeated in the same exact way without the dropped columns

DF1
str(DF1)
DF2<-DF1[,-c(3,7,8,11,13:16)]
str(DF2)
set.seed(4)
indexes<-sample(1:nrow(DF2),as.integer(nrow(DF2)/5),replace=FALSE)
TestDF2<-DF2[indexes,]
TrainDF2<-DF2[-indexes,]
Trainlabels2<-TrainDF2$SalaryLabel
Testlabels2<-TestDF2$SalaryLabel
TrainDF2<-TrainDF2[,-which(names(TrainDF2) %in% c('Salary_pct'))]
TestDF2<-TestDF2[,-which(names(TestDF2) %in% c('SalaryLabel','Salary_pct'))]
(model2<-naiveBayes(TrainDF2,Trainlabels2,laplace = 1))
pred2<-predict(model2,TestDF2)
cm2<-confusionMatrix(pred2,Testlabels2)
cmDF2<-as.data.frame(cm2$table)
ggplot(data=cmDF2,aes(x=Reference,y=Prediction))+
  geom_tile(aes(fill=Freq))+
  geom_text(aes(label = round((Freq/sum(Freq)),2))) +
  scale_fill_gradient(low='blue',high='red')+
  theme_bw()+
  xlim(rev(levels(cmDF$Reference)))
plot_confusion_matrix(cmDF2, 
                      target_col = "Reference", 
                      prediction_col = "Prediction",
                      counts_col = "Freq",
                      add_row_percentages = FALSE,
                      add_col_percentages = FALSE,
                      rm_zero_percentages = FALSE,
                      rm_zero_text = FALSE,
                      add_zero_shading = TRUE,
                      counts_on_top = TRUE,
                      palette='Blues') 
        