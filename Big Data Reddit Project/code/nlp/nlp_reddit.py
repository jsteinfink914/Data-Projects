# Databricks notebook source
# MAGIC %md
# MAGIC ## Business Question:
# MAGIC `Business Goal:` Identify the categories of emergencies that are most prevalent in medical questions. This information is relevant because the majority of healthcare costs come from patients admitted in emergency rooms. Identifying the most common reasons for these emergencies would help implement targeted prevention measures to reduce these emergencies, while also reducing healthcare costs imposed on the healthcare system.
# MAGIC 
# MAGIC `Technical Proposal:` Using the text in the **emergencymedicine**, **AskDocs**, and **medicine** subreddits, we will be able to examine causes for medical concern. This can be accomplished by applying regex and creating Boolean variables. In order to find the relevant diagnoses, we will join the subreddit data with external data that provides a list of emergency room diagnoses (this data can be found in *data/csv/emergency_room_diagnoses_2021.csv*). New Boolean variables can then be added to determine which of these diagnoses can be identified in the subreddits, thus identifying if the concern usually ends up in the ER. We can also use sentiment analysis to determine relative severity of these diagnoses and compare. Additionally, by performing NLP and examining common words we can gain insight into important context around these concerns.

# COMMAND ----------

## IMPORTING ALL THE LIBRARIES
import os
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql.functions import col, isnan, when, count, regexp_extract, split, col, lower, regexp_replace
import pyspark.sql.functions as f
from pyspark.sql.types import ArrayType, DoubleType
import re
from pyspark.ml.feature import CountVectorizer, HashingTF, IDF
from pyspark.ml import Pipeline
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import StringType, MapType
from pyspark.sql import Window

# COMMAND ----------

## Setting up file directories to save small data and associated plots
CSV_DIR = "../../data/csv"
PLOT_DIR = "../../data/plots"

# COMMAND ----------

ems_sub = spark.read.parquet("/FileStore/ems_subs2")
ems_com = spark.read.parquet("/FileStore/ems_coms2")

# COMMAND ----------

ems_sub.printSchema()

# COMMAND ----------

ems_com.printSchema()

# COMMAND ----------

print(ems_sub.count())

# COMMAND ----------

print(ems_com.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading in External Emergency Room Diagnoses

# COMMAND ----------

diagnoses = spark.read.parquet("/FileStore/erd_2021").toPandas()

# COMMAND ----------

diagnoses.head()

# COMMAND ----------

top_diagnoses = diagnoses.groupby('DiagnosisDesc').sum('TotalDiag').sort_values(by = ['TotalDiag'], ascending = False).head(75)

# COMMAND ----------

# ------------------------------------------------------------------------------------------------------------------------------------ #
#
# REDEFINE NAMES FOR THE DIAGNOSES TO FIT REDDIT DISCOURSE
#
# ------------------------------------------------------------------------------------------------------------------------------------ #

new_diags = ['digestive tract', 'upper respiratory infection', 'allergy to narcotic', 'allergy to antibiotic', 'allergy to drugs', 'allergy to penicillin', 'anemia', 'anxiety', 'atherosclerosis', 'coronary artery heart disease', 'chest pain', 'chronic obstructive pulmonary disease', 'constipation', 'exposure to COVID-19', 'cough', 'COVID-19', 'dehydration', 'diarrhea', 'dizziness', 'immunization', 'stomach pain', 'hypertension', 'fever', 'gastro-esophageal reflux disease', 'headache', 'heart failure', 'homelessness', 'hyperlipidemia', 'hypothyroidism', 'anticoagulants', 'aspirin', 'insulin', 'hypoglycemic', 'lower back pain', 'depression', 'nausea', 'nausea with vomiting', 'addiction to nicotine', 'addiction', 'obesity', 'other chest pain', 'other chronic pain', 'drug therapy', 'postprocedural', 'nicotine dependence', 'transient ischemic attack', 'hypercholesterolemia', 'shortness of breath', 'collapse', 'tachycardia', 'type 2 diabetes with chronic kidney disease', 'type 2 diabetes', 'abdominal pain', 'uncomplicated asthma', 'atrial fibrillation', 'head injury', 'urinary tract infection', 'unspecified vomiting', 'weakness']


# COMMAND ----------

# ------------------------------------------------------------------------------------------------------------------------------------ #
# 
# IDENTIFY IF ANY WORDS IN THE TEXT ARE IN THE LIST OF DIAGNOSES
#
# ------------------------------------------------------------------------------------------------------------------------------------ #
diagnoses_str = "(?i)" + "|(?i)".join(new_diags)
ems_sub_df = ems_com.withColumn('diagnosed', when(col('body').rlike(diagnoses_str), True))

# COMMAND ----------

# ------------------------------------------------------------------------------------------------------------------------------------ #
#
# IF SO, THEN ADD THE MATCHED WORD TO A COLUMN CALLED "EMS_TOPIC" 
#
# ------------------------------------------------------------------------------------------------------------------------------------ #
ems_labeled = ems_sub_df.withColumn('ems_topic', regexp_extract(col('body'), diagnoses_str, 0))
ems_labeled.show()

# COMMAND ----------

# ------------------------------------------------------------------------------------------------------------------------------------ #
# 
# ADD DUMMY FOR FATAL OR NOT
# 
# ------------------------------------------------------------------------------------------------------------------------------------ #
death_diags = ['death', 'died', 'die', 'fatal', 'fatality', 'deceased', 'mortality', 'mortal', 'dying', 'passed away', 'passing away', 'killed', 'lost his life', 'lost her life', 'loss of life', 'flatline', 'pulled the plug']

death_str = "(?i)" + "|(?i)".join(death_diags)
ems_labeled = ems_labeled.withColumn('fatal', when(col('body').rlike(death_str), True))

ems_labeled.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Table 1

# COMMAND ----------

## Displaying a table of counts of comments that mention ER diagnoses and those that do not
diagnosed_count = ems_labeled.groupBy(col('diagnosed')).count().toPandas()
diagnosed_count

# COMMAND ----------

# MAGIC %md
# MAGIC This table shows the number of comments we found mentioning an ER diagnosis vs. not mentioning one. Out of 1.7 million comments across the 3 subreddits, around 74,000 mentioned some form of common ER diagnosis.

# COMMAND ----------

diagnosed_count.to_csv(os.path.join(CSV_DIR,"diagnosis_mentions_count.csv"))

# COMMAND ----------

all_topics = ems_labeled.where(col('ems_topic') != "")
all_topics.show()

# COMMAND ----------

# all_topics.write.parquet('/FileStore/Medical_topics_data')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Skip to here to read in Merged Data

# COMMAND ----------

topics_labeled = spark.read.parquet('/FileStore/Medical_topics_data')
topics_labeled = topics_labeled.withColumn('ems_topic', f.lower(col('ems_topic')))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Building Sentiment Model

# COMMAND ----------

# ------------------------------------------------------------------------------------------------------------------------------------ #
#
# CLEAN BODY OF DATA
#
# ------------------------------------------------------------------------------------------------------------------------------------ #
# Clean the text of unnecessary characters and such
def clean_text(c):
  c = lower(c)
  c = regexp_replace(c, "(https?\://)\S+", "") # Remove links
  c = regexp_replace(c, "(\\n)|\n|\r|\t", "") # Remove CR, tab, and LR
  c = regexp_replace(c, "(?:(?:[0-9]{2}[:\/,]){2}[0-9]{2,4})", "") # Remove dates
  c = regexp_replace(c, "@([A-Za-z0-9_]+)", "") # Remove usernames
  c = regexp_replace(c, "[0-9]", "") # Remove numbers
  c = regexp_replace(c, "\:|\/|\#|\.|\?|\!|\&|\"|\,", "") # Remove symbols
  return c
df = topics_labeled.withColumn("body", clean_text(col("body"))) # Clean the body column

# Document assembler to prep the body of the reddit posts
document = DocumentAssembler()\
    .setInputCol("body")\
    .setOutputCol("document")

# Sentence Detector
sentence = SentenceDetector()\
    .setInputCols(['document'])\
    .setOutputCol('sentence')

# Tokenizer
token = Tokenizer()\
    .setInputCols(['sentence'])\
    .setOutputCol('token')

# Stop words
stop_words = StopWordsCleaner.pretrained('stopwords_en', 'en')\
    .setInputCols(["token"]) \
    .setOutputCol("cleanTokens") \
    .setCaseSensitive(False)

tokenAssembler = TokenAssembler() \
    .setInputCols(["sentence", "cleanTokens"]) \
    .setOutputCol("cleanText")

use = UniversalSentenceEncoder.pretrained(name="tfhub_use", lang="en")\
 .setInputCols(["cleanText"])\
 .setOutputCol("sentence_embeddings")


sentimentdl = SentimentDLModel.pretrained(name='sentimentdl_use_twitter', lang="en")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("sentiment")

prediction_pipeline = Pipeline(
    stages = [
        document,
        sentence,
        token,
        stop_words,
        tokenAssembler,
        use,
        sentimentdl
    ]
)

result = prediction_pipeline.fit(df).transform(df)


# COMMAND ----------

sentiment_df = result.select('body', 'retrieved_on','subreddit','ems_topic',f.explode('sentiment.result'))
sentiment_df.show(5)

# COMMAND ----------

## Writing to dbfs
#sentiment_df.write.parquet("/FileStore/sentiment_ems")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Skip to here to read in Sentiment Data

# COMMAND ----------

sentiment_df = spark.read.parquet('/FileStore/sentiment_ems')
sentiment_df = sentiment_df.withColumnRenamed("col","sentiment")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Summary Graph of Counts of Each Diagnosis

# COMMAND ----------

# ------------------------------------------------------------------------------------------------------------------------------------ #
#
# CREATE SUMMARY TABLE OF COUNTS OF EACH DIAGNOSIS
#
# ------------------------------------------------------------------------------------------------------------------------------------ #
# This answers part of the question "What are the most common words overall?".
ems_com_summary = sentiment_df.groupby('ems_topic').count().toPandas()
ems_com_summary['ems_topic'] = ems_com_summary['ems_topic'].str.upper()

ems_com_summary_sorted = ems_com_summary.sort_values(by = ['count'], ascending = False).reset_index().drop(columns = ['index'])
ems_com_summary_sorted

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Graph 1

# COMMAND ----------

## Creating plot of top 20 diagnoses
top_20_diagnosis = ems_com_summary_sorted.head(20).sort_values(by = ['count'])
plt.barh(y = 'ems_topic', width = 'count', data = top_20_diagnosis)
plt.xlabel("Count")
plt.ylabel("Diagnosis Mentioned")
plt.title("Most Frequently mentioned ER Diagnoses on Reddit")
plt.savefig(os.path.join(PLOT_DIR, "top_20_diagnosis.jpg"))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The plot (Plot 1) above illustrates the counts of common emergency room diagnoses that were the subject of Emergency Medicine, AskDocs, and medicine subreddits. Based on the data, the most prevalent reason for serious medical discussion is anxiety, with 17,416 mentions. Fever is a relatively close second, while the rest of the mentions surprisingly have to do with general health and not acute events. Thus, the table above confirms that the majority of hospital visits are actually a result of bad health, as opposed to disastrous events.   

# COMMAND ----------

## Saving the csv
top_20_diagnosis.to_csv(os.path.join(CSV_DIR, "top_20_diagnosis.csv"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### What is the distribution of text lengths?

# COMMAND ----------

# ------------------------------------------------------------------------------------------------------------------------------------ #
#
# What is the distribution of text lengths? (of the posts with diagnosis key words, how long is each post)
#
# ------------------------------------------------------------------------------------------------------------------------------------ #

length_df = sentiment_df.withColumn("length", f.length(col('body')))


# COMMAND ----------

# MAGIC %md
# MAGIC #### Table 2

# COMMAND ----------

length_summary = length_df.agg(  
                            f.mean(col('length')).alias('mean'),
                            f.expr("percentile(length, array(0.25))")[0].alias('Q25'),
                            f.expr('percentile(length, array(0.50))')[0].alias('median'),
                            f.expr('percentile(length, array(0.75))')[0].alias('Q75')).toPandas().rename(index = {0:"length"})
length_summary

# COMMAND ----------

length_summary.to_csv(os.path.join(CSV_DIR, "ER_com_length_dist.csv"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Graph 2 

# COMMAND ----------

l = length_df.groupBy(col('ems_topic')).agg(f.mean('length').alias("mean_length")).toPandas()

# COMMAND ----------

top_20_length = l.sort_values(by = ['mean_length'], ascending = False).head(20).sort_values(by = ['mean_length'])
y = top_20_length['ems_topic'].str.upper().tolist()
x = top_20_length['mean_length'].tolist()
plt.barh(y = y, width = x)
plt.xlabel("Mean Length")
plt.ylabel("Diagnosis Mentioned")
plt.title("Avg. Comment Length by ER Diagnoses on Reddit (Top 20)")
plt.savefig(os.path.join(PLOT_DIR, "top_20_diagnosis_length.jpg"))
plt.show()

# COMMAND ----------

top_20_length.to_csv(os.path.join(CSV_DIR, "top_20_diagnosis_length.csv"))

# COMMAND ----------

# MAGIC %md
# MAGIC Graph 2 shows how a lot of the most known and serious issues generate the most thoughtful replies. Posprocedural is an interesting leader, but given the lack of common discussion around it and complications that ensue, it is an area that many people clearly have thoughts on. We also see drug related topics, Diabetes, and COVID-19 on this list indicating that people like to talk about common yet serious problems at length.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sentiment Model Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC #### Table 3

# COMMAND ----------

## Examining % of positive to negative comments
sentiment_dist = sentiment_df.groupBy('sentiment').count().toPandas()
sentiment_dist['percent'] = round(100 * sentiment_dist['count']/sentiment_dist['count'].sum(),2)
sentiment_dist = sentiment_dist[['sentiment','percent']]
sentiment_dist

# COMMAND ----------

sentiment_dist.to_csv(os.path.join(CSV_DIR, "sentiment_dist.csv"))

# COMMAND ----------

# MAGIC %md
# MAGIC The majority of the comments mentioning an ER diagnosis are negative which makes sense given the serious and often grave nature of a lot of these ailments. The positive comments are likely hopeful responses to people's concerns, which is nice to see.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Graph 3 - TF-IDF Graph

# COMMAND ----------

# Dev for TF-IDF

#REFERENCE: https://stackoverflow.com/questions/69218494/pyspark-display-top-10-words-of-document 

# Prep the documents to pass to the TF-IDF
# Used the cleaned, tokenized data that we created from the pipeline
documents = result.select("*").withColumn("label", monotonically_increasing_id())
documents = documents.select("label","cleanTokens.result")
#documents.show(2)

# #hashingTF way
hashingTF = HashingTF(inputCol="result", outputCol="rawFeatures")
featurizedData = hashingTF.transform(documents)
#featurizedData.show(5)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
# rescaledData.show(truncate=False)
#print(hashingTF.indexOf("Possibly"))


## to read output vector and map to word, we use hash values of each word using same fitted model and map with features vector indices and get corresponding values.
ndf = documents.select('label',f.explode('result').name('expwords')).withColumn('result',f.array('expwords'))
#ndf.show(5) # cols: "label", "expwords", "words"
hashudf = f.udf(lambda vector : vector.indices.tolist()[0],StringType())
wordtf = hashingTF.transform(ndf).withColumn('wordhash',hashudf(f.col('rawFeatures')))
#wordtf.show()

## flatten output features column to get indices & value.
udf1 = f.udf(lambda vec : dict(zip(vec.indices.tolist(),vec.values.tolist())),MapType(StringType(),StringType()))
valuedf = rescaledData.select('label',f.explode(udf1(f.col('features'))).name('wordhash','value'))
#valuedf.show()

## get top n words for each document(label) filtering based on its rank and join both DFs and collect & sort to get the words along with its value.
w = Window.partitionBy('label').orderBy(f.desc('value'))
valuedf = valuedf.withColumn('rank',f.rank().over(w)).where(f.col('rank')<=5) # used 5 for testing.

topn_df = valuedf.join(wordtf,
                       ['label','wordhash']).groupby('label').agg(f.sort_array(f.collect_set(f.struct(f.col('value'),f.col('expwords'))),asc=False).name('topn')) #.show(truncate=False)

topn_df.printSchema()


# COMMAND ----------

#https://sparkbyexamples.com/pyspark/pyspark-select-nested-struct-columns/
words_and_vals = topn_df.select("topn.value", "topn.expwords") #.show(truncate = False)
words_and_vals.show()

# COMMAND ----------

from pyspark.sql.functions import explode

# GOAL: Try to explode each row so that each word is associated with a specific value and a label that connects the word & value to the corresponding row in  words_and_vals 
wv_labeled = words_and_vals.select("*").withColumn("row_label", monotonically_increasing_id()) # label the rows to identify when words come from different comments

# COMMAND ----------

wv_exploded_expwords = wv_labeled.select(wv_labeled.row_label, explode(wv_labeled.expwords)).withColumn("index", monotonically_increasing_id()).selectExpr("index", "row_label", "col as word")
wv_exploded_values = wv_labeled.select(wv_labeled.row_label, explode(wv_labeled.value)).withColumn("index", monotonically_increasing_id()).selectExpr("index", "row_label as row_label_1", "col as value")

wv_exploded = wv_exploded_expwords.join(wv_exploded_values, on = 'index')
wv_exploded_final = wv_exploded.selectExpr("index", "row_label", "word", "value")
wv_exploded_final.show(10)

# COMMAND ----------

from pyspark.sql.functions import sum,avg,count

# Now creating a graph for word importance  

# group by the word and take the average of the values if there are multiple
avg_count = wv_exploded_final.groupBy('word').agg(avg('value').alias('average'),
                                                   count('word').alias('count'))
avg_count.show(10)


# COMMAND ----------


# convert to Pandas dataframe
avg_count_df = avg_count.toPandas() # convert to dataframe


# COMMAND ----------

# convert to csv
# avg_count_df.to_csv(os.path.join(CSV_DIR, "avg_count.csv"))

# COMMAND ----------

import numpy as np

# THIS MAY NOT GIVE AN ACCURATE REPRESENTATION OF THE DATA, IF IT'S CLEANED IN THIS WAY
# find the outliers and remove them

q1 = np.quantile(avg_count_df['average'], 0.25)
q3 = np.quantile(avg_count_df['average'], 0.75)

iqr = q3 - q1

lower = q1 - (1.5 * iqr)
upper = q3 + (1.5 * iqr)

remove_lower = avg_count_df[avg_count_df['average'] >= lower]
avg_count_df_cleaned = remove_lower[remove_lower['average'] <= upper]


avg_count_df_cleaned = avg_count_df_cleaned.reset_index(drop = True)
avg_count_df_cleaned

# convert to csv
avg_count_df_cleaned.to_csv(os.path.join(CSV_DIR, "avg_count_cleaned.csv"))

# COMMAND ----------

## top 20 words plotted
top_20_tfidf = avg_count_df_cleaned.sort_values('average', ascending = False)[:20]

plt.barh(y = "word", width = "average", data = top_20_tfidf)
plt.title("Results of TF-IDF Analysis")
plt.ylabel('Word')
plt.xlabel('TF-IDF')
plt.savefig(os.path.join(PLOT_DIR, "top_20_tfidf.jpg"))
plt.show()

## next idea: have the above graph without words overlayed (and a trend line instead) to visualize the relationship btwn the count and avg importance of all the words

# COMMAND ----------

# MAGIC %md
# MAGIC #### Graph 4/5 - Most Positive and Most Negative ER Diagnoses

# COMMAND ----------

import numpy as np
ratios = sentiment_df.withColumn("pos", (col("sentiment") == "positive").cast('int')).withColumn("neg", (col("sentiment") == "negative").cast('int')).withColumn("neutral", (col("sentiment") == "neutral").cast('int'))
r = ratios.groupBy(col('ems_topic')).agg(f.mean("pos").alias("pos"), f.mean("neg").alias("neg"), f.mean("neutral").alias("neutral")).toPandas()
r['pos_neg'] = r['pos']/r['neg']
r['neg_pos'] = r['neg']/r['pos']
r['ems_topic'] = r['ems_topic'].str.upper()
top_20_neg = r.replace([np.inf], 0).sort_values(by = 'neg_pos', ascending = False).head(20).sort_values(by = 'neg_pos')
top_20_pos = r.replace([np.inf], 0).sort_values(by = 'pos_neg', ascending = False).head(20).sort_values(by = 'pos_neg')

# COMMAND ----------

plt.barh(y = "ems_topic", width = "neg_pos", data = top_20_neg)
plt.title("Most Negative ER Diagnoses (Top 20)")
plt.ylabel('Diagnosis')
plt.xlabel('Avg. Neg:Pos Ratio')
plt.savefig(os.path.join(PLOT_DIR, "top_20_neg_diagnoses.jpg"))
plt.show()

# COMMAND ----------

plt.barh(y = "ems_topic", width = "pos_neg", data = top_20_pos)
plt.title("Most Positive ER Diagnoses (Top 20)")
plt.ylabel('Diagnosis')
plt.xlabel('Avg. Pos:Neg Ratio')
plt.savefig(os.path.join(PLOT_DIR, "top_20_pos_diagnoses.jpg"))
plt.show()

# COMMAND ----------

## Save the csv's
top_20_neg.to_csv(os.path.join(CSV_DIR, "top_20_neg_diagnoses.csv"))
top_20_pos.to_csv(os.path.join(CSV_DIR, "top_20_pos_diagnoses.csv"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Business Question 2: (Added question to aid with Deliverable #3)
# MAGIC `Business Goal`: Understand how similar Pro-Life and Pro-Choice supporters are. Is it possible to find middle group or are these groups wildly different. Understanding how similar or different these groups are can either deny or confirm prior held beliefs about the relationship between the two groups. This would allow companies, politicians, etc. to tailor their campaigns and advertising accordingly.
# MAGIC 
# MAGIC `Technical Proposal`: Using the text in the ProLife and ProChoice subreddits, we will perform data cleaning and analysis as defined in the pipeline above. New Boolean variables are also added to determine if any other keywords ('abortion', 'guns', 'religion', etc.) are used. We can also use sentiment analysis to determine relative severity of these posts and compare. 

# COMMAND ----------

# Read the Reproductive Rights dataset. Contains data from 3 subreddits: roevwade2022, prolife, and prochoice.
# Read the data
RR_sub = spark.read.parquet("/FileStore/ReproductiveRightsv2_subs")
RR_com = spark.read.parquet("/FileStore/ReproductiveRightsv2_coms")

# COMMAND ----------

import matplotlib.pyplot as plt
from pyspark.sql.functions import col,isnan, when, count
import pyspark.sql.functions as f

# Let's create new columns for certain controversial topics/words
# First, let's only work with the relevant content of the posts with title and body.
RR_sub_df = RR_sub.select("subreddit", "title", f.lower("title"))
RR_com_df = RR_com.select("subreddit", "body", f.lower("body"))

# Topic: Abortion
# This is a major proponent of reproductive rights and we would expect the frequency of this word to appear frequently
RR_sub_df = RR_sub_df.withColumn("Abortion", col('lower(title)').rlike("(?i)abortion"))
RR_com_df = RR_com_df.withColumn("Abortion", col('lower(body)').rlike("(?i)abortion"))

# Topic: Guns
# Though this has nothing to do with reproductive rights, this word has appeared throughout the subreddits
RR_sub_df = RR_sub_df.withColumn("Guns", col('lower(title)').rlike("(?i)gun|(?i)guns|(?i)nra|(?i)firearms|(?i)firearm"))
RR_com_df = RR_com_df.withColumn("Guns", col('lower(body)').rlike("(?i)gun|(?i)guns|(?i)nra|(?i)firearms|(?i)firearm"))

# Topic: Religion
# Healthcare and religion are often very interwined in today's world
RR_sub_df = RR_sub_df.withColumn("Religion", col('lower(title)').rlike("(?i)religion|(?i)god|(?i)sin|(?i)jesus|(?i)holy|(?i)pray|(?i)sinner|(?i)hell|(?i)devil|(?i)church|(?i)bible"))
RR_com_df = RR_com_df.withColumn("Religion", col('lower(body)').rlike("(?i)religion|(?i)god|(?i)sin|(?i)jesus|(?i)holy|(?i)pray|(?i)sinner|(?i)hell|(?i)devil|(?i)church|(?i)bible"))

# Topic: Politics (general terms)
# Reproductive rights has been heavily contested in the political arena
RR_sub_df = RR_sub_df.withColumn("Politics", col('lower(title)').rlike("(?i)politics|(?i)political|(?i)party|(?i)law|(?i)legal|(?i)illegal|(?i)voting|(?i)vote|(?i)campaign|(?i)election|(?i)media|(?i)democracy|(?i)lawmaker|(?i)politicians|(?i)legalize|(?i)legalized|(?i)outlaw|(?i)outlawed"))
RR_com_df = RR_com_df.withColumn("Politics", col('lower(body)').rlike("(?i)politics|(?i)political|(?i)party|(?i)law|(?i)legal|(?i)illegal|(?i)voting|(?i)vote|(?i)campaign|(?i)election|(?i)media|(?i)democracy|(?i)lawmaker|(?i)politicians|(?i)legalize|(?i)legalized|(?i)outlaw|(?i)outlawed"))

# Topic: Democrat (Party Specific mentions)
RR_sub_df = RR_sub_df.withColumn("Democrat", col('lower(title)').rlike("(?i)democrat|(?i)liberal|(?i)biden|(?i)kamala|(?i)harris|(?i)nancy|(?i)pelosi"))
RR_com_df = RR_com_df.withColumn("Democrat", col('lower(body)').rlike("(?i)democrat|(?i)liberal|(?i)biden|(?i)kamala|(?i)harris|(?i)nancy|(?i)pelosi"))

# Topic: Republicam (Party Specific mentions)
RR_sub_df = RR_sub_df.withColumn("Republican", col('lower(title)').rlike("(?i)republican|(?i)conservative|(?i)trump|(?i)mcconnell"))
RR_com_df = RR_com_df.withColumn("Republican", col('lower(body)').rlike("(?i)republican|(?i)conservative|(?i)trump|(?i)mcconnell"))

# Topic: RoeVWade (Specific mention of the ruling)
# The overturning of roe v. wade acted as a catalyst for increase in engagement on their subreddits.
RR_sub_df = RR_sub_df.withColumn("RoeVWade", col('lower(title)').rlike("(?i)roe|(?i)wade|(?i)supreme court|(?i)overruled|(?i)overrule|(?i)overturned|(?i)overturn|(?i)ruling|(?i)neil gorsuch|(?i)brett kavanaugh|(?i)amy coney barrett"))
RR_com_df = RR_com_df.withColumn("RoeVWade", col('lower(body)').rlike("(?i)roe|(?i)wade|(?i)supreme court|(?i)overruled|(?i)overrule|(?i)overturned|(?i)overturn|(?i)ruling|(?i)neil gorsuch|(?i)brett kavanaugh|(?i)amy coney barrett"))

# COMMAND ----------

# MAGIC %md
# MAGIC We're going to redo the steps from the pipeline above. So let's repeat.

# COMMAND ----------

# Clean the body of the comments data
# FUNCTION: clean_text defined in the code above
df = RR_com_df.withColumn("body", clean_text(col("body"))) # Clean the body column

# COMMAND ----------

#df.take(5)

# COMMAND ----------

# Run our cleaned body data through the pre-defined pipeline
# PIPELINE: defined in the code above
result = prediction_pipeline.fit(df).transform(df)

# COMMAND ----------

sentiment_df = result.select('body','subreddit', 'Abortion', 'Guns', 'Religion', 'Politics', 'Democrat', 'Republican', 'RoeVWade', f.explode('sentiment.result'))
sentiment_df.show(5)

# COMMAND ----------

## Writing to dbfs
sentiment_df.write.parquet("/FileStore/sentiment_rr_v2") # Reproductive Rights sentiment data

# COMMAND ----------


