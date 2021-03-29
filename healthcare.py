from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Basics').getOrCreate()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pylab import rcParams
from pyspark.sql.functions import avg
import pyspark.sql.functions as f
from pyspark.sql.window import Window #***
from pyspark.sql.functions import sum
from pyspark.sql.functions import format_number

#dataframe of healthcare
df = spark.read.csv('/home/giotanna/Documents/DM_project_2021/healthcare-dataset-stroke-data/healthcare-dataset-stroke-data.csv', inferSchema=True,header=True)

df.show()

df.describe().show()

df.printSchema()

df.groupBy('gender').mean().show()

#gender(Male, Female, Other) that has a stroke
df.groupBy('gender','stroke').mean().show()

#how many people have or not a stroke
df.groupBy('stroke').count().show()

#prople have stroke depended from work
#more are private
df_stroke = df.groupBy('work_type', 'stroke').count().select('stroke','work_type',f.col('count').alias('work_type_count'))
df_stroke = df_stroke.filter(df['stroke'] == 1).select('work_type', 'work_type_count').orderBy(df_stroke['work_type_count'].desc())
df_stroke.show()

#participated in this clinic measurement
#58% Female. 41% Male
df_mes = df.groupBy('gender').count().select('gender',f.col('count').alias('count_gender')).withColumn('percentage_of_gender',(f.col('count_gender') / df.count()) * 100 ).select('gender','count_gender',(format_number('percentage_of_gender',2).alias('percentage_of_gender')))
df_mes.show()


# how many female/male have a stroke
# 2,11% Male. 2,76% Female
df_gen_str_M = df.groupBy('gender', 'stroke').count().select('gender',f.col('count').alias('count_gender')).withColumn('percentage_of_stroke',(f.col('count_gender') / df.count()) * 100 ).filter((df['stroke'] == 1) & (df['gender'] == 'Male')).select('gender','count_gender',(format_number('percentage_of_stroke',2).alias('percentage_of_stroke')))
df_gen_str_M.show()
df_gen_str_F = df.groupBy('gender', 'stroke').count().select('gender',f.col('count').alias('count_gender')).withColumn('percentage_of_stroke',(f.col('count_gender') / df.count()) * 100 ).filter((df['stroke'] == 1) & (df['gender'] == 'Female')).select('gender','count_gender',(format_number('percentage_of_stroke',2).alias('percentage_of_stroke')))
df_gen_str_F.show()

df.distinct().show()

#sort by age 
df_ag = df.groupBy('age','stroke').count().select('age', 'stroke', 'count')
df_ag.show()

#sort by stroke and age > 50
df_age = df.groupBy('age','stroke').count().select('stroke','age',f.col('count').alias('age_count')).withColumn('percentage_of_stroke',(f.col('age_count') / df.count()) * 100 ).filter(df['stroke'] == 1).select('age','age_count',(format_number('percentage_of_stroke',2).alias('percentage_of_stroke')))
df_age = df_age.orderBy(df_age['age_count'].desc())
df_age.show()

df2= spark.sql("SELECT df.age FROM df GROUP BY df.age ")
df2.show()

# calculate the number of stroke cases for people after 50 years 
df.filter((df['stroke'] == 1) & (df['age'] > '50')).count()

#percentage person over 50 and stroke
#4.4 %
df_age_per = ((df.filter((df['stroke'] == 1) & (df['age'] > '50')).count() ) / df.count() ) * 100 
df_age_per