##Importing Necessary libraries
library(arules)
library(arulesViz)
library(dplyr)
library(twitteR)
library(ROAuth)
library(ggplot2)
library(tokenizers)
library(stopwords)
library(rtweet)
library(plyr)
library(stringr)
library(rtweet)
library(networkD3)
library(kableExtra)
library(igraph)
library(visNetwork)
##Converting to market basket format
Trans<-read.transactions('AssociationData.csv',format='basket',sep=',')
inspect(Trans)
summary(Trans)
##Using apriori algorithm 
Rules<-arules::apriori(Trans,parameter=list(support=.15,confidence=.2,
                                            minlen=2))
inspect(Rules)

##Sorting rules by support
SortedRules_Sup<-sort(Rules,by='support',decreasing=TRUE)
##Sorting by confidence
SortedRules_conf<-sort(Rules,by='confidence',decreasing=TRUE)
##Sorting by Lift
SortedRules_lift<-sort(Rules,by='lift',decreasing=TRUE)

##Plotting table using kable package and styling
kbl(inspect(SortedRules_Sup[1:15]))  %>% kable_classic(lightable_options = 'striped')
kbl(inspect(SortedRules_conf[1:15])) %>% kable_classic(lightable_options = 'striped')
kbl(inspect(SortedRules_lift[1:15])) %>% kable_classic(lightable_options = 'striped')

##Plotting interactive graph
plot(SortedRules_Sup[1:15],measure='support',shading='lift',engine='interactive',
     method='graph')


##Using visNetwork

##BUilding Nodes and edges
##Using aviz DATAFRAME function
Rules_DF<-DATAFRAME(SortedRules_conf[1:15])
##Isolating nodes 
Rules_DF$LHS<-as.character(Rules_DF$LHS)
Rules_DF$RHS<-as.character(Rules_DF$RHS)
##Removing {}
Rules_DF[]<-lapply(Rules_DF,gsub,pattern='[{]',replacement='')
Rules_DF[]<-lapply(Rules_DF,gsub,pattern='[}]',replacement='')
edges1<-Map(c,Rules_DF$LHS,Rules_DF$RHS)
edges2<-purrr::flatten(edges1)
edges3<-as.character(edges2)

##PLotting igraph Graph
plot(graph(edges=edges3))

##Building edgeList nodesList and plotting with vizNetwork
Rules_DF<-DATAFRAME(SortedRules_lift[1:15])
##Isolating nodes 
Rules_DF$LHS<-as.character(Rules_DF$LHS)
Rules_DF$RHS<-as.character(Rules_DF$RHS)
##Removing {}
Rules_DF[]<-lapply(Rules_DF,gsub,pattern='[{]',replacement='')
Rules_DF[]<-lapply(Rules_DF,gsub,pattern='[}]',replacement='')
edges1<-Map(c,Rules_DF$LHS,Rules_DF$RHS)
edges2<-purrr::flatten(edges1)
edges3<-as.character(edges2)
edgeList<-data.frame(from=Rules_DF$LHS,to=Rules_DF$RHS,label=as.character(round(as.numeric(Rules_DF$lift),2)))
nodesList<-data.frame(id=unique(edges3),label=unique(edges3))
visNetwork(nodes=nodesList,edges=edgeList)
