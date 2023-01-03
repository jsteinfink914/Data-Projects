
# Setup -------------------------------------------------------------------
rm(list=ls())
library(nflreadr)
library(dplyr)
library(BSDA)
library(ggplot2)
library(gridExtra)
library(cowplot)
library(forcats)


# Data Gathering ----------------------------------------------------------

##Collecting play by play data for every year from 2012-2021. 2011 was when the
##kickoff rules changed

years<-2012:2021
DF<-data.frame()
for (i in years){
   DF<-rbind(DF,load_pbp(i))
}

## Filtering to select only Onisde kicks and determine if they were recovered
Onside<- filter(DF,play_type=='kickoff' & grepl('onside|Onside|ONSIDE', desc)) %>%
  mutate(year = 
           as.numeric(substr(as.character(game_id),0,4)),
         Recovered = 
           ifelse((grepl('Recovered|RECOVERED',desc) & !grepl('REVERSED',desc)),1,0)) %>%
  group_by(year) %>%
  mutate(conversion = mean(Recovered)) 

# Saving data
write.csv(Onside,'data/Onside.csv',row.names=F)



# Onside Kick Analysis ----------------------------------------------------------------

Onside_pre_2018<-filter(Onside,year<2018)
Onside_Post_2018<-filter(Onside,year>2017)

##Check to see if onside kick percentages are different now that teams 
##cant get a running start
mean(Onside_pre_2018$Recovered)
sd_x<-sd(Onside_pre_2018$Recovered)
mean(Onside_Post_2018$Recovered)
sd_y<-sd(Onside_Post_2018$Recovered)
z.test(x=Onside_pre_2018$Recovered,y=Onside_Post_2018$Recovered,sigma.x=sd_x,sigma.y=sd_y)

##Making histogram plot with a normal distribution drawn on top
numbers<-rnorm(10000)
y<-as.data.frame(numbers)
ggplot(y,aes(x=numbers))+
  geom_histogram(color='darkblue',fill='darkblue',aes(y =after_stat(density)))+
  ##Drawing the normal distribution on top of the histogram
  stat_function(fun = dnorm, args = list(mean = 0, sd = 1))+
  ##Labeling the is_west coefficient
  geom_text(x = 1.7166,y=.2, label = "Post_2018 vs. Pre_2018",size=7,vjust=-.5,hjust=0,angle=90)+
  geom_vline(xintercept = 1.7166, linetype='dashed', color = 'red')+
  geom_vline(xintercept=1.96,color='orange')+
  geom_text(x=1.96,y=.35,label='95% Cutoff Threshold',size=7,vjust=-1,hjust=0,angle=270)+
  xlab('SD')+
  ggtitle('Comparison of Pre and Post 2018 Onside Kick Conversion Rates')+
  theme(plot.title=element_text(vjust=1,size=20,hjust=.5),axis.title.x = element_text(size=15),
        axis.title.y = element_text(size=15),panel.background = element_blank(),panel.grid.major.y=element_line(colour='grey'))

counts<-
  Onside %>% 
  group_by(year) %>% 
  summarise(count_1=n()) %>%
  as.data.frame()

ggplot(counts,aes(x=year,y=count_1))+
  geom_bar(stat='identity',fill='darkblue')+
  geom_vline(xintercept = 2017.5,linetype='dashed',color='red')+
  geom_text(aes(x=2017.5, label="\nRules Changed", y=55), colour="red",angle=90, size= 7)+
  ggtitle('Onside Kick Attempts Over Time')+
  ylab('Count')+
  xlab('Year')+
  theme(plot.title=element_text(vjust=1,size=20,hjust=.5),axis.title.x = element_text(size=15),
        axis.title.y = element_text(size=15),axis.text.x = element_text(size=13), axis.text.y= element_text(size=13),panel.background = element_blank(),panel.grid.major.y=element_line(colour='grey'))


# 4th Down Analysis -------------------------------------------------------


Fourth <- filter(DF,down==4)
#Fourth_any<-filter(Fourth,play_type=='pass' | play_type=='run')
Fourth_5 <- filter(Fourth,ydstogo>5 & ydstogo<=10 & (play_type=='pass' | play_type=='run'))
Fourth_10 <- filter(Fourth,ydstogo>10 & ydstogo<=15 & (play_type=='pass' | play_type=='run'))
Fourth_15<- filter(Fourth,ydstogo>15 & ydstogo<=20 & (play_type=='pass' | play_type=='run'))
Fourth_20 <- filter(Fourth,ydstogo>20 & ydstogo<=25 & (play_type=='pass' | play_type=='run'))

F_list<-list(Fourth_5,Fourth_10,Fourth_15,Fourth_20)

f<-mean(Fourth_5$first_down)
t<-mean(Fourth_10$first_down)
ft<-mean(Fourth_15$first_down)
tt<-mean(Fourth_20$first_down)

percents<-c(f,t,ft,tt)
f_counts<-as.numeric(as.character(lapply(F_list,nrow)))
names_f<-c('Fourth_5-10','Fourth_10-15','Fourth_15-20','Fourth_20-25')

Fourth_Conversion<-data.frame(names_f,percents,f_counts)
Fourth_Conversion<-
  Fourth_Conversion %>%
  mutate(name=fct_reorder(names_f,desc(f_counts)))

Rate<-ggplot(Fourth_Conversion,aes(x=name,y=percents))+
  geom_bar(stat='identity',fill='darkblue',colour='orange')+
  ggtitle('Fourth Down Conversion Rates')+
  xlab('Categories')+
  ylab('Conversion %')+
  theme(plot.title=element_text(vjust=1,size=20,hjust=.5),axis.title.x = element_text(size=15),
        axis.title.y = element_text(size=15),axis.text.x = element_text(size=15), axis.text.y= element_text(size=15),panel.background = element_blank(),panel.grid.major.y=element_line(colour='grey'))
  
Hist<-ggplot(Fourth_Conversion,aes(x=name,y=f_counts))+
  geom_bar(stat='identity',fill='darkblue',colour='orange')+
  ggtitle('Fourth Down Attempts')+
  xlab('Categories')+
  ylab('Counts')+
  theme(plot.title=element_text(vjust=1,size=20,hjust=.5),axis.title.x = element_text(size=15),
        axis.title.y = element_text(size=15),axis.text.x = element_text(size=15), axis.text.y= element_text(size=15),panel.background = element_blank(),panel.grid.major.y=element_line(colour='grey'))

plot_grid(Rate,Hist)


F_15<-z.test(x=Onside_Post_2018$Recovered,y=Fourth_15$first_down,sigma.x=sd(Onside_Post_2018$Recovered),sigma.y=sd(Fourth_15$first_down))
F_20<-z.test(x=Onside_Post_2018$Recovered,y=Fourth_20$first_down,sigma.x=sd(Onside_Post_2018$Recovered),sigma.y=sd(Fourth_20$first_down))
F_10<-z.test(x=Onside_pre_2018$Recovered,y=Fourth_10$first_down,sigma.x=sd(Onside_pre_2018$Recovered),sigma.y=sd(Fourth_10$first_down))
F_5 <-z.test(x=Onside_pre_2018$Recovered,y=Fourth_5$first_down,sigma.x=sd(Onside_pre_2018$Recovered),sigma.y=sd(Fourth_5$first_down))

ggplot(y,aes(x=numbers))+
  geom_histogram(fill='darkblue',aes(y =..density..))+
  ##Drawing the normal distribution on top of the histogram
  stat_function(fun = dnorm, args = list(mean = 0, sd = 1))+
  ##Labeling the is_west coefficient
  geom_vline(xintercept=F_20$statistic, linetype='dashed',color='red')+
  geom_text(x=F_20$statistic,y=.1,label='Fourth + 20-25',size=7,vjust=-1,hjust=0,angle=90,color='white')+
  geom_vline(xintercept=F_15$statistic, linetype='dashed',color='red')+
  geom_text(x=F_15$statistic,y=.10,label='Fourth + 15-20',size=7,vjust=-1,hjust=0,angle=90, color='black')+
  geom_vline(xintercept=1.96, color='orange')+
  geom_text(x=1.96,y=.375,label='95% Cutoff Threshold',size=7,vjust=-1,hjust=0,angle=270)+
  geom_vline(xintercept=-1.96,color='orange')+
  geom_text(x=-1.96,y=.225,label='95% Cutoff Threshold',size=7,vjust=-1,hjust=0,angle=90)+
  xlab('SD')+
  ggtitle('Null Hypothesis: Fourth + 20 Conversion Rate = Post 2018 Onside Kick Conversion Rate')+
  theme(plot.title=element_text(vjust=1,size=20,hjust=.5),axis.title.x = element_text(size=15),
        axis.title.y = element_text(size=15),axis.text.y = element_text(size=15), axis.text.x = element_text(size=15),panel.background = element_blank(),panel.grid.major.y=element_line(colour='grey'))
