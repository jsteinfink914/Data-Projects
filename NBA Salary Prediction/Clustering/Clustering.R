###Clustering NBA Record Data
##This code looks to validate the salary labels as well as explore the data
##and see if positions can roughly be clustered based on specific stats

##Importing necessary libraries
library(dplyr)
library(NbClust)
library(cluster)
library(mclust)
library(factoextra)
library(akmeans)
library(stylo)
library(fpc)
library(dbscan)
library(ggplot2)
library(plotly)
library(caret)
library(patchwork)
library(gridExtra)
library(cowplot)
norm_DF<-read.csv('NormalizedDF_Labels.csv')
str(norm_DF)
## Select only players with over 15 games Played which is 
## 15/83=.18
DF1<-subset(norm_DF,games_played>=.18)
##Change to Numeric and remove labels and non-numeric columns
##Isolate Salary label
label<-as.factor(DF1$SalaryLabel)
##Remove non-numeric columns
DF<-DF1[,-c(20:26)]
##Convert to numeric
DF<-DF %>% mutate_all(as.numeric)
str(DF)

##Clustering Efficiency, true shooting att, and 2k Ratings
##to try and validate salary labels
DFcluster3<- subset(DF) %>% select(efficiency,Ratings_2k,
                                   Salary_pct)
(DFCluster_Eucl <- dist(DFcluster3,method="euclidean", p=2))  
(DFCluster_Man <- dist(DFcluster3,method="manhattan"))
(DFCluster_Cos <- stylo::dist.cosine(as.matrix(DFcluster3)))

##Using histogram and hclust to see clusters
Hist1<-hclust(DFCluster_Eucl,method='ward.D2')
Hist2 <- hclust(DFCluster_Man, method="ward.D2")
Hist3 <- hclust(DFCluster_Cos, method="ward.D2")
##Plotting the dendrograms in one image
par(mfrow=c(1,3))
plot(Hist1,labels=FALSE,col='blue',main='Euclidean Dendrogram')
plot(Hist2,labels=FALSE,col='red',main='Manhattan Dendrogram')
plot(Hist3,labels=FALSE,col='green',main='Cosine Dendrogram')
##hclust seems to suggest a range of clusters from 2-8
##Check silhouette elbow and gap stat methods using all 3 distance
##metrics

##ELBOW METHOD
##Euclidean
Euclidean_elbow<-fviz_nbclust(DFcluster3, FUN = hcut, method = "wss", 
                              k.max = 8) +
                  ggtitle("WSS:Elbow:Euclidean")
##Manhattan
Manhattan_elbow<-fviz_nbclust(DFcluster3, FUN = hcut, method = "wss",
                              diss=DFCluster_Man, k.max = 8) +
                  ggtitle("WSS:Elbow:Manhattan")
##Cosine
Cosine_elbow<-fviz_nbclust(DFcluster3, FUN = hcut, method = "wss",
                           diss=DFCluster_Cos, k.max = 8) +
                  ggtitle("WSS:Elbow:Cosine")

##Display graphs in one image using gridExtra package

plot_grid(Euclidean_elbow,Manhattan_elbow,Cosine_elbow,nrow=1)

##ELbow method loosely suggests 3 clusters but there is a viable range 
##of up to 8

###SILHOUETTE
##Euclidean
Euc_sil<-fviz_nbclust(DFcluster3, FUN = hcut, method = "silhouette", 
             k.max = 8) +
  ggtitle("Silhouette:Euclidean")
##Manhattan
Man_sil<-fviz_nbclust(DFcluster3, FUN = hcut, method = "silhouette", 
             diss=DFCluster_Man,k.max = 8) +
  ggtitle("Silhouette:Manhattan")
##Cosine
Cos_sil<-fviz_nbclust(DFcluster3, FUN = hcut, method = "silhouette", 
             diss=DFCluster_Cos,k.max = 8) +
  ggtitle("Silhouette:Cosine")
##Plotting them together
plot_grid(Euc_sil,Man_sil,Cos_sil,nrow=1)

##Silhouette method suggesting 2 clusters but viable range of up to 8

###GAP STATISTIC
##Euclidean
Euc_gap<-fviz_nbclust(DFcluster3, FUN = hcut, method = "gap_stat", 
                      diss=DFCluster_Eucl,k.max = 8) +
  ggtitle("Gap Stat:Euclidean")
##Manhattan
Man_gap<-fviz_nbclust(DFcluster3, FUN = hcut, method = "gap_stat", 
             diss=DFCluster_Man, k.max = 8) +
  ggtitle("Gap Stat:Manhattan")
##Cosine
Cos_gap<-fviz_nbclust(DFcluster3, FUN = kmeans, method = "gap_stat", 
             diss=DFCluster_Cos, k.max = 8) +
  ggtitle("Gap Stat:Cosine")

##Plotting
plot_grid(Euc_gap,Man_gap,Cos_gap,nrow=1)
###Suggests 4 clusters but 7 and 8 are also viable

##Kmeans Clustering with Euclidean
k<-c(4,7,8)
for (i in k){
  kmeansResult<-kmeans(DFcluster3,i)
  plot(fviz_cluster(kmeansResult,data=DFcluster3,geom = c('point','text'),
                    axes=c(1,3),main=(paste('Euclidean Kmeans',i,'clusters')),
                    ggtheme=theme_minimal()))
  ##Creating 3D Viz
  if (i==4){
    kmeansDF<-DFcluster3
    kmeansDF$cluster<-kmeansResult$cluster
    figure<-plot_ly(x=kmeansDF$efficiency,y=kmeansDF$Ratings_2k,z=kmeansDF$Salary_pct,
                    type='scatter3d',mode='markers',color=kmeansDF$cluster)
    figure<-figure%>%layout(scene=list(xaxis=list(title='PER'),
                                       yaxis=list(title='2k Rating'),
                                       zaxis=list(title='Salary%')))
    
  }
}
figure
##Use confusionMatrix to check accuracy
##Reformat label to be in same format
##Labels may have to be changed depending on what 
##clusters represent which data
for (i in 1:length(kmeansDF$cluster)){
  if (kmeansDF$cluster[i]==4){
    kmeansDF$cluster[i]<-'below average'
  }
  else if(kmeansDF$cluster[i]==2){
    kmeansDF$cluster[i]<-'average'
  }
  else if (kmeansDF$cluster[i]==1){
    kmeansDF$cluster[i]<-'star'
  }
  else if (kmeansDF$cluster[i]==3){
    kmeansDF$cluster[i]<-'superstar'
  }
}
##Plotting frequency of cluster results
ggplot(kmeansDF,aes(x=cluster))+geom_bar(fill='red')+ggtitle('Cluster Bins')
results<-as.factor(kmeansDF$cluster)
confusionMatrix(results,label)

##Kmeans with Manhattan
for (i in k){
  kmeansResult<-pam(DFcluster3,i,metric='manhattan')
  plot(fviz_cluster(kmeansResult,data=DFcluster1,geom = c('point','text'),axes=c(1,3),
                    main=(paste('Manhattan Kmeans',i,'clusters')),ggtheme=theme_minimal()))
  if (i==4){
    kmeansDF<-DFcluster3
    kmeansDF$cluster<-kmeansResult$cluster
    figure<-plot_ly(x=kmeansDF$efficiency,y=kmeansDF$Ratings_2k,z=kmeansDF$Salary_pct,
                    type='scatter3d',mode='markers',color=kmeansDF$cluster)
    figure<-figure%>%layout(scene=list(xaxis=list(title='PER'),
                                       yaxis=list(title='2k Ratings'),
                                       zaxis=list(title='Salary%')))
    
  }
}
figure
##Use confusionMatrix to check accuracy
##Reformat label to be in same format
##Labels may have to be changed depending on what 
##clusters represent which data
for (i in 1:length(kmeansDF$cluster)){
  if (kmeansDF$cluster[i]==4){
    kmeansDF$cluster[i]<-'below average'
  }
  else if(kmeansDF$cluster[i]==3){
    kmeansDF$cluster[i]<-'average'
  }
  else if (kmeansDF$cluster[i]==1){
    kmeansDF$cluster[i]<-'star'
  }
  else if (kmeansDF$cluster[i]==2){
    kmeansDF$cluster[i]<-'superstar'
  }
}
ggplot(kmeansDF,aes(x=cluster))+geom_bar(fill='red')+ggtitle('Cluster Bins')
results<-as.factor(kmeansDF$cluster)
confusionMatrix(results,label)

##Kmeans with Cosine
for (i in k){
  kmeansResult<-pam(as.matrix(DFCluster_Cos),i)
  plot(fviz_cluster(kmeansResult,data=DFcluster1,geom = c('point','text'),axes=c(1,3),
                    main=(paste('Cosine Kmeans',i,'clusters')),ggtheme=theme_minimal()))
}
##DBSCAN
db <- fpc::dbscan(DFcluster3, eps = 0.05, MinPts = 5)
# Plot DBSCAN results
fviz_cluster(db, DFcluster3, main = "DBSCAN", geom='point')

