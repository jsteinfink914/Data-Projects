library(ggplot2)

##Changing the salary labels to be more specific 
DF<-read.csv('Clean_Data_Labels.csv')
DF<-subset(DF,games_played>=15)
summary(DF$Salary_pct)
##Subsetting the data for those who only played over 15 games removes
##many outliers that would complicate the analysis
##Using the updated median to mark the below average category
##Median to 3rd quartile marks above average
##Median to 90th percentile marks the star category
##Top 10% are superstars

indexes1<-which(DF[,'Salary_pct']<=3.87,arr.ind=T)
indexes2<-which(DF[,'Salary_pct']>3.87 & DF[,'Salary_pct']<=10.09,arr.ind=T)
indexes3<-which(DF[,'Salary_pct']>10.09 & DF[,'Salary_pct']<=19.17,arr.ind=T)
indexes4<-which(DF[,'Salary_pct']>19.17,arr.ind=T)
##ASsigning names based on description above
DF[indexes1,'SalaryLabel']<-'below average'
DF[indexes2,'SalaryLabel']<-'average'
DF[indexes3,'SalaryLabel']<-'star'
DF[indexes4,'SalaryLabel']<-'superstar'
DF$SalaryLabel<-as.factor(DF$SalaryLabel)
##Visualizing the bins
ggplot(DF,aes(SalaryLabel))+
  ggtitle('Salary Bins')+
  geom_bar(fill='red')

write.csv(DF,'Clean_Data_UpdatedLabels.csv',row.names = FALSE)
DF1<-read.csv('NormalizedDF_Labels.csv')
DF1$SalaryLabel<-DF1$SalaryLabel
write.csv(DF1,'NormalizedDF_UpdatedLabels.csv',row.names = FALSE)



