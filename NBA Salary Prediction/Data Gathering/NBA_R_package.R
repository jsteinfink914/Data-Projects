##Insalling and loading the nbastatR package
devtools::install_github('abresler/nbastatR')
library(nbastatR)
##Setting the connection size to double the default to support the data intake
Sys.setenv("VROOM_CONNECTION_SIZE"=131072*2)
##Using the R package to collect All NBA team rosters
All_NBA<-all_nba_teams()
##Writing the results to a csv file
write.csv(All_NBA,'All_NBA.csv',row.names = FALSE)


