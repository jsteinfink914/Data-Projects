##Importing necessary libraries
library(arules)
library(arulesViz)
library(dplyr)
library(ggplot2)

##Read in the data
DF<-read.csv('Clean_Data.csv')
str(DF)

##Remove unnecessary columns
DF<-subset(DF,select=-c(year,name,full_name,All_Nba,All_Nba_team))
##Removing players with less than 15 games played
DF1<-subset(DF,games_played>=15)
DF1<-DF1[,-1]

##Have to Bin All the data, will do this by quartiles
##Bottom = 1st quartile 
##Low=2nd quartile
##high= 3rd quartile
##Top= 4th quartile

for (i in names(DF1)){
  ##Collecting quartiles for each column
  quartile1<-as.numeric(quantile(DF1[,i],.25))
  quartile2<-median(DF1[,i])
  quartile3<-as.numeric(quantile(DF1[,i],.75))
  ##locating indexes of the different quartiles
  indexes1<-which(DF1[,i]<=quartile1,arr.ind=T)
  indexes2<-which(DF1[,i]>quartile1 & DF1[,i]<=quartile2,arr.ind=T)
  indexes3<-which(DF1[,i]>quartile2 & DF1[,i]<=quartile3,arr.ind=T)
  indexes4<-which(DF1[,i]>quartile3,arr.ind=T)
  ##ASsigning names based on description above
  DF1[indexes1,i]<-paste('Bottom',i)
  DF1[indexes2,i]<-paste('Low',i)
  DF1[indexes3,i]<-paste('High',i)
  DF1[indexes4,i]<-paste('Top',i)
  
}

##Writing results to csv file
write.csv(DF1,'AssociationData.csv',row.names=FALSE)
