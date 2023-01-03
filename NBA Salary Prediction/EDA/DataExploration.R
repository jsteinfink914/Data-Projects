library(ggplot2)
library(reshape2)
DF<-read.csv('Clean_Data.csv')
##Reformat tprimary_positon, All_NBA and All_Nba_team
DF$primary_position<-as.factor(DF$primary_position)
DF$All_Nba<-as.factor(DF$All_Nba)
DF$All_Nba_team<-as.factor(DF$All_Nba_team)
hist(DF$Salary_pct,main='Normalized Salary',xlab='Salary%', col='orange')
##Making bins for Salary %

# Given the distribution of salary, binning is a good approach to take. For more context on the variable of Salary as a percentage of the Salary Cap, the current 
# cap in the NBA is around $112 million. Thus a salary of 10% represents an $11.2 million contract in 2020, well above the mean of $8.32 million in 2020, and 
# above the mean of $6.77 million in the dataset. To keep the bins general at first, 4 seperate tiers will be made: below average, above average, star, and superstar. These bins
# will reflect different levels of player salaries. The below average bin will include players who make below the average (below 6.77 million). The above average bin will 
# include players who make anywhere from the mean to the 75th percentile of salary (6.77%-9.25%) reflecting salaries from $6.77 million to around $10.5 million (in 2020 terms). 
# The star's bin will have salaries from the 75th percentile to the 90th percentile (9.25%-18.59%) reflecting salaries from $10.5 million to $21 million. The last bin, superstar,
# will include the top 10% of NBA earners.
for (i in 1:length(DF$Salary_pct)){
  if (DF[i,"Salary_pct"]<=3.41){
    DF[i,'SalaryLabel']<-'below average'
  }
  else if(DF[i,"Salary_pct"]<=9.25){
    DF[i,'SalaryLabel']<-'average'
  }
  else if (DF[i,"Salary_pct"]<=18.59){
    DF[i,'SalaryLabel']<-'star'
  }
  else if (DF[i,"Salary_pct"]>18.59){
    DF[i,'SalaryLabel']<-'superstar'
  }
}
DF$SalaryLabel<-as.factor(DF$SalaryLabel)
##Writing the new dataset to a csv
write.csv(DF,'Clean_Data_Labels.csv',row.names = FALSE)
##Writing labels to the normalized data frame
DF1<-read.csv('NormalizedDF.csv')
DF1$SalaryLabel<-DF$SalaryLabel
write.csv(DF1,'NormalizedDF_Labels.csv',row.names=FALSE)
##Visualizing the bins
ggplot(DF,aes(SalaryLabel))+
  ggtitle('Salary Bins')+
  geom_bar(fill='red')
  labs(x='',y="")
  

##Exploring the data

boxplot(Salary_pct~All_Nba,data=DF,
        col=(c("gold","darkgreen")),
        main="All NBA vs. non All NBA salaries", xlab="All NBA status",
        ylab='Salary %')
boxplot(Salary_pct~primary_position,data=DF,
        col=(c("blue","darkgreen",'red','purple','gold')),
        main="Salary by position", xlab="Position",ylab='Salary %')
boxplot(Salary_pct~SalaryLabel,data=DF,
        col=(c("gold","grey",'lightblue','green')),
        main="Salary by Label", xlab="Label",ylab='Salary %')
boxplot(Ratings_2k~primary_position,data=DF,
        col=(c("blue","darkgreen",'red','purple','gold')),
        main="2k ratings by position", xlab="Position",ylab='2k Rating')
boxplot(Salary_pct~All_Nba_team,data=DF,
        col=(c("gold","darkgreen")),
        main="Salaries by All NBA team", xlab="All NBA team",
        ylab='Salary %')
##The biggest outlier in Salary for those who did not make an All NBA team
##for that year...Kobe Bryant in 2013

boxplot(efficiency~primary_position,data=DF,
        col=(c("gold","darkgreen")),
        main="Efficiency by Postion", xlab="Position",
        ylab='PER')
boxplot(true_shooting_pct~primary_position,data=DF,
        col=(c("gold","darkgreen")),
        main="TS% by postion", xlab="Position",
        ylab='PER')

cormat<-round(cor(DF[,-c(1:4,20,21,26)],DF[,-c(1:4,20,21,26)],method='spearman', use = "pairwise.complete.obs"),2)
get_lower_tri<-function(cormat){
  cormat[upper.tri(cormat)] <- NA
  return(cormat)
}
ggplot(melt(get_lower_tri(cormat),na.rm=T), aes(Var1, Var2, fill=value)) +
  geom_tile(height=1, width=1,colour='black') +
  scale_fill_gradient2(low="blue", mid="white", high="red") +
  geom_text(aes(Var1, Var2, label = value), color = "black", size = 3) +
  theme_minimal() +
  ggtitle('Correlation Matrix')+
  coord_equal() +
  labs(x="",y="",fill="Corr") +
  theme(axis.text.x=element_text(size=10, angle=45, vjust=1, hjust=1, 
                                 margin=margin(-3,0,0,0)),
        axis.text.y=element_text(size=10, margin=margin(0,-3,0,0)),
        panel.grid.major=element_blank()) 

boxplot(assists_turnover_ratio~primary_position,data=DF,
        col=(c("blue","darkgreen",'red','purple','gold')),
        main="Assist to TO ratio by position", xlab="Position",
        ylab='Assists/Turnovers')
##Outlier for shooting guards is a player who played 8 games
##May want to think about dropping rows of players who did not
##play enough games


