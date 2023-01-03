##Importing necessary libraries
library(rtweet)
library(ROAuth)
library(jsonlite)
library(base64enc)

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
##Performing a series of searches based on key terms and hashtags
##Writing the outuput of each to a distinct file

Search<-rtweet::search_tweets('nba allstar',n=1000, lang = 'en',
                              include_rts = FALSE,retryonratelimit = TRUE)
filename<-'Twitter_text_files/nba_allstar.txt'
MyFile<- file(filename)
cat(unlist(Search$text)," ",file=MyFile,sep='\n')
close(MyFile)


Search<-rtweet::search_tweets('nba star',n=1000,lang='en',include_rts = FALSE,retryonratelimit = TRUE)
filename<-'Twitter_text_files/nba_star.txt'
MyFile<- file(filename)
cat(unlist(Search$text)," ",file=MyFile,sep='\n')
close(MyFile)


Search<-rtweet::search_tweets('nba mvp',n=1000,lang='en',include_rts = FALSE,retryonratelimit = TRUE)
filename<-'Twitter_text_files/nba_mvp.txt'
MyFile<- file(filename)
cat(unlist(Search$text)," ",file=MyFile,sep='\n')
close(MyFile)

Search<-rtweet::search_tweets('#AllNBA',n=1000,lang='en',include_rts = FALSE,retryonratelimit = TRUE)
filename<-'Twitter_text_files/#AllNBA.txt'
MyFile<- file(filename)
cat(unlist(Search$text)," ",file=MyFile,sep='\n')
close(MyFile)

Search<-rtweet::search_tweets('nba superstar',n=1000,lang='en',include_rts = FALSE,retryonratelimit = TRUE)
filename<-'Twitter_text_files/nba_superstar.txt'
MyFile<- file(filename)
cat(unlist(Search$text)," ",file=MyFile,sep='\n')
close(MyFile)

Search<-rtweet::search_tweets('nba legend',n=1000,lang='en',include_rts = FALSE,retryonratelimit = TRUE)
filename<-'Twitter_text_files/nba_legend.txt'
MyFile<- file(filename)
cat(unlist(Search$text)," ",file=MyFile,sep='\n')
close(MyFile)

