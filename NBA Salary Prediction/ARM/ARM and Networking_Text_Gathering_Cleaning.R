##Libraries
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

##Setting Twitter authorization keys
consumer_key<-'klef9rsiaHCjKuU0RJUKTPsY9'
consumer_secret<- 'yZyTEAm70kPYsFo1X2xmQR1wc87Eavapnfhp9rp8jIMIUMgxzh'
access_token<- '1409604349538492419-nEOthOigmEiE5lGUjOkHj7JWTkPOrG'
access_secret<- 'lmUKRmRb0VT09LIZr4mlUptx7vGAyPLtGHYvxgGzVPYJy'
##Accessing the twitter api directly 
twitter_token <- create_token(
  consumer_key = consumer_key,
  consumer_secret = consumer_secret,
  access_token = access_token,
  access_secret = access_secret)
##Setting a masterile to store all searches and cluster1 file to 
##store allstar, legend, superstar searches
Masterfile<-'Twitter_csv_files/all_searches.csv'
Cluster1<-'Twitter_csv_files/allstar_legend_superstar.csv'

##Performing a series of searches based on key terms and
##Writing the outuput of each to a distinct file
Search<-rtweet::search_tweets('nba allstar',n=1000, lang = 'en',
                              include_rts = FALSE,retryonratelimit = TRUE)
Search_DF<-as.data.frame(Search)
filename<-'Twitter_csv_files/nba_allstar.csv'
MyFile<-file(filename)
File1<-file(Cluster1)
File2<-file(Masterfile)
for (i in 1:nrow(Search_DF)){
  ##Tokenizing words and removing unnecessary numbers,punctuation, ans 
  ##stop words
  Tokens<-tokenizers::tokenize_words(Search_DF$text[i],
                                     stopwords=stopwords::stopwords('en'),
                                     lowercase=TRUE,strip_punct=TRUE,
                                     strip_numeric=TRUE,simplify=TRUE)
  ##Writing words to csv files
  cat(unlist(str_squish(Tokens)),'\n',file=filename,sep=',',append=TRUE)
  cat(unlist(str_squish(Tokens)),'\n',file=Cluster1,sep=',',append=TRUE)
  cat(unlist(str_squish(Tokens)),'\n',file=Masterfile,sep=',',append=TRUE)
}
close(MyFile)
close(File1)
close(File2)



Search<-rtweet::search_tweets('nba star',n=1000,lang='en',include_rts = FALSE,
                              retryonratelimit = TRUE)
filename<-'Twitter_csv_files/nba_star.csv'
MyFile<-file(filename)
File2<-file(Masterfile)
Search_DF<-as.data.frame(Search)
for (i in 1:nrow(Search_DF)){
  Tokens<-tokenizers::tokenize_words(Search_DF$text[i],
                                     stopwords=stopwords::stopwords('en'),
                                     lowercase=TRUE,strip_punct=TRUE,
                                     strip_numeric=TRUE,simplify=TRUE)
  cat(unlist(str_squish(Tokens)),'\n',file=filename,sep=',',append=TRUE)
  cat(unlist(str_squish(Tokens)),'\n',file=Masterfile,sep=',',append=TRUE)
}
close(MyFile)
close(File2)


Search<-rtweet::search_tweets('nba mvp',n=1000,lang='en',include_rts = FALSE,
                              retryonratelimit = TRUE)
filename<-'Twitter_csv_files/nba_mvp.csv'
MyFile<-file(filename)
File2<-file(Masterfile)
Search_DF<-as.data.frame(Search)
for (i in 1:nrow(Search_DF)){
  Tokens<-tokenizers::tokenize_words(Search_DF$text[i],
                                     stopwords=stopwords::stopwords('en'),
                                     lowercase=TRUE,strip_punct=TRUE,
                                     strip_numeric=TRUE,simplify=TRUE)
  cat(unlist(str_squish(Tokens)),'\n',file=filename,sep=',',append=TRUE)
  cat(unlist(str_squish(Tokens)),'\n',file=Masterfile,sep=',',append=TRUE)
}
close(MyFile)
close(File2)


Search<-rtweet::search_tweets('nba superstar',n=1000,lang='en',
                              include_rts = FALSE,retryonratelimit = TRUE)
filename<-'Twitter_csv_files/nba_superstar.csv'
MyFile<-file(filename)
File1<-file(Cluster1)
File2<-file(Masterfile)
Search_DF<-as.data.frame(Search)
for (i in 1:nrow(Search_DF)){
  Tokens<-tokenizers::tokenize_words(Search_DF$text[i],
                                     stopwords=stopwords::stopwords('en'),
                                     lowercase=TRUE,strip_punct=TRUE,
                                     strip_numeric=TRUE,simplify=TRUE)
  cat(unlist(str_squish(Tokens)),'\n',file=filename,sep=',',append=TRUE)
  cat(unlist(str_squish(Tokens)),'\n',file=Cluster1,sep=',',append=TRUE)
  cat(unlist(str_squish(Tokens)),'\n',file=Masterfile,sep=',',append=TRUE)
}
close(MyFile)
close(File1)
close(File2)

Search<-rtweet::search_tweets('nba legend',n=1000,lang='en',include_rts = FALSE,
                              retryonratelimit = TRUE)
filename<-'Twitter_csv_files/nba_legend.csv'
MyFile<-file(filename)
Search_DF<-as.data.frame(Search)
File1<-file(Cluster1)
File2<-file(Masterfile)
for (i in 1:nrow(Search_DF)){
  Tokens<-tokenizers::tokenize_words(Search_DF$text[i],
                                     stopwords=stopwords::stopwords('en'),
                                     lowercase=TRUE,strip_punct=TRUE,
                                     strip_numeric=TRUE,simplify=TRUE)
  cat(unlist(str_squish(Tokens)),'\n',file=filename,sep=',',append=TRUE)
  cat(unlist(str_squish(Tokens)),'\n',file=Cluster1,sep=',',append=TRUE)
  cat(unlist(str_squish(Tokens)),'\n',file=Masterfile,sep=',',append=TRUE)
}
close(MyFile)
close(File1)
close(File2)


##Cleaning the CSV files
Allstar_legend_superstar<-read.csv('Twitter_csv_files/allstar_legend_superstar.csv',header=FALSE)
All_searches<-read.csv('Twitter_csv_files/all_searches.csv',header=FALSE)
Allstar<-read.csv('Twitter_csv_files/nba_allstar.csv',header=FALSE)
Legend<-read.csv('Twitter_csv_files/nba_legend.csv',header=FALSE)
MVP<-read.csv('Twitter_csv_files/nba_mvp.csv',header=FALSE)
star<-read.csv('Twitter_csv_files/nba_star.csv',header=FALSE)
superstar<-read.csv('Twitter_csv_files/nba_star.csv',header=FALSE)
#Storing all the files in a list so they can be looped through
DataList<-list(Allstar_legend_superstar,All_searches,
               Allstar,Legend,MVP,star,superstar)
##Empty list to store cleaned Data
NewDataList<-list()
##Removing specific words that are unnecessary and show up often
for (i in 1:length(DataList)){
  DF<-as.data.frame(DataList[[i]])
  ##Setting empty dataframes
  No_numbers<-NULL
  Short_words<-NULL
  Long_words<-NULL
  ##Replacing 4 unnecessary words
  DF[DF=='t.co']<-''
  DF[DF=='rt']<-''
  DF[DF=='http']<-''
  DF[DF=='https']<-''
  ##Looping through each column
  for (a in 1:ncol(DF)){
    ##lists to be filled with logicals
    numbers<-c()
    numbers<-c(numbers,grepl("[[:digit:]]",DF[[a]])) ##removing digits
    shortwords<-c()
    shortwords<-c(shortwords,grepl("[A-z]{4,}",DF[[a]]))##for short words
    longwords<-c()
    longwords<-c(longwords,grepl("[A-z]{12,}",DF[[a]]))##for long words
    ##Appending lists to create dataframes full of logicals
    No_numbers<-cbind(No_numbers,numbers)
    Short_words<-cbind(Short_words,shortwords)
    Long_words<-cbind(Long_words,longwords)
  }
  ##Replacing 'appropriate'TRUE' entries with blanks
  DF[No_numbers]<-''
  DF[!Short_words]<-''
  DF[Long_words]<-''
  NewDataList[[i]]<-DF
}
##Reassingning names to the clean DataFrames
Allstar_legend_superstar<-NewDataList[[1]]
All_searches<-NewDataList[[2]]
Allstar<-NewDataList[[3]]
Legend<-NewDataList[[4]]
MVP<-NewDataList[[5]]
star<-NewDataList[[6]]
superstar<-NewDataList[[7]]

##Writing to csv files
write.csv(Allstar_legend_superstar,'Twitter_csv_files/allstar_legend_superstar_clean.csv',row.names = FALSE)
write.csv(All_searches,'Twitter_csv_files/all_searches_clean.csv',row.names = FALSE)
write.csv(Allstar,'Twitter_csv_files/nba_allstar_clean.csv',row.names = FALSE)
write.csv(Legend,'Twitter_csv_files/nba_legend_clean.csv',row.names = FALSE)
write.csv(MVP,'Twitter_csv_files/nba_mvp_clean.csv',row.names = FALSE)
write.csv(star,'Twitter_csv_files/nba_star_clean.csv',row.names = FALSE)
write.csv(superstar,'Twitter_csv_files/nba_superstar_clean.csv',row.names = FALSE)




