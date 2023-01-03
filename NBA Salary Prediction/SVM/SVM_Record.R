##Importing necessary libraries
library(naivebayes)
library(ggplot2)
library(e1071)
library(dplyr)
library(caret)
library(cvms)
library(rsvg)
library(ggimage)
library(kableExtra)
library(rminer)
library(reshape2)
library(plotly)

##Importing data
DF<-read.csv('Clean_Data_UpdatedLabels.csv')
str(DF)
##Removing non-numeric columns
DF<-DF[,-c(1:4)]
##Remove players who played less than 15 games 
DF<-filter(DF,DF$games_played>15)
DF<-filter(DF,)
DF$SalaryLabel<-as.factor(DF$SalaryLabel)

##Normalizing the data
min_max_norm <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}
NormalizedDF<-as.data.frame(lapply(DF[,-c(22)],
                                   function(x) min_max_norm(x)))
DF1<-NormalizedDF
DF1$SalaryLabel<-DF$SalaryLabel
DF<-NormalizedDF
DF$SalaryLabel<-DF1$SalaryLabel
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
TrainDF<-TrainDF[,-which(names(TrainDF) %in% c('Salary_pct'))]
TestDF<-TestDF[,-which(names(TestDF) %in% c('SalaryLabel','Salary_pct'))]

##Running Linear SVM model
tuned_cost<-tune(svm,SalaryLabel~.,data=TrainDF,kernel='linear',ranges=
                   list(cost=c(.01,.1,1,10,100,1000)))
summary_linear<-summary(tuned_cost)
kbl(summary_linear$performances) %>% kable_material_dark(lightable_options ='striped')
(model<-svm(SalaryLabel~.,data=TrainDF,kernel = 'linear',cost = 100,probability=TRUE))


pred<-predict(model,TestDF)

##Plotting 3d viz using 2k Ratings, efficiency, and true shooting attempts
##with predicted salary labels

figure<-plot_ly(x=TestDF$efficiency,y=TestDF$Ratings_2k,z=TestDF$true_shooting_att,
                type='scatter3d',mode='markers',color=pred)
figure<-figure%>%layout(title='Linear SVM',
                        scene=list(xaxis=list(title='PER'),
                                   yaxis=list(title='2k Rating'),
                                   zaxis=list(title='TS%')))
figure
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

##Running Radial SVM model
tuned_cost2<-tune(svm,SalaryLabel~.,data=TrainDF,kernel='radial',ranges=
                    list(cost=c(.01,.1,1,10,100,1000)))
summary_radial<-summary(tuned_cost2)
kbl(summary_radial$performances) %>% kable_material_dark(lightable_options ='striped')
(model2<-svm(SalaryLabel~.,data=TrainDF,kernel='radial',cost=10))
pred2<-predict(model2,TestDF)

##Plotting 3d viz using 2k Ratings, efficiency, and true shooting attempts
##with predicted salary labels

figure2<-plot_ly(x=TestDF$efficiency,y=TestDF$Ratings_2k,z=TestDF$true_shooting_att,
                type='scatter3d',mode='markers',color=pred2)
figure2<-figure2%>%layout(title='Radial SVM',
                        scene=list(xaxis=list(title='PER'),
                                   yaxis=list(title='2k Rating'),
                                   zaxis=list(title='TS%')))
figure2
##Creating confusion matrix
cm2<-confusionMatrix(pred2,Testlabel)
cmDF2<-as.data.frame(cm2$table)
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

##Running Polynomial SVM model
tuned_cost3<-tune(svm,SalaryLabel~.,data=TrainDF,kernel='polynomial',ranges=
                    list(cost=c(.01,.1,1,10,100,1000)))
summary_polynomial<-summary(tuned_cost3)
kbl(summary_polynomial$performances) %>% kable_material_dark(lightable_options ='striped')
(model3<-svm(SalaryLabel~.,data=TrainDF,kernel = 'polynomial',cost=10,probability=TRUE))
pred3<-predict(model3,TestDF)

##Plotting 3d viz using 2k Ratings, efficiency, and true shooting attempts
##with predicted salary labels

figure3<-plot_ly(x=TestDF$efficiency,y=TestDF$Ratings_2k,z=TestDF$true_shooting_att,
                type='scatter3d',mode='markers',color=pred3)
figure3<-figure3%>%layout(title='Poly SVM',
                        scene=list(xaxis=list(title='PER'),
                                   yaxis=list(title='2k Rating'),
                                   zaxis=list(title='TS%')))
figure3
##Creating confusion matrix
cm3<-confusionMatrix(pred3,Testlabel)
cmDF3<-as.data.frame(cm3$table)
plot_confusion_matrix(cmDF3, 
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

##Transfer labels into numbers 
x<-melt(TrainDF,measure.vars = names(TrainDF))
y<-as.factor(Trainlabels)
z<-as.vector(as.matrix(model3$decision.values))
data<-as.data.frame(x$value,y,z,row.names = 1:53160)
