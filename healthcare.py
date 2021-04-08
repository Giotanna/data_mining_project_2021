from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import OneHotEncoder
from pyspark.sql.types import FloatType, StructField, StructType, IntegerType, StringType

spark = SparkSession.builder.getOrCreate()
healthcare_df = (
    spark.read.option("delimiter", ",").csv("healthcare-dataset-stroke-data.csv", header=True, inferSchema=True)
    # .filter(F.col("gender") != "Other")
)

healthcare_df.printSchema()

schema = StructType(
    [
        StructField("id", IntegerType()),
        StructField("gender", StringType()),
        StructField("avg_glucose_level", FloatType()),
        StructField("bmi", FloatType()),
        StructField("smoking_status", StringType()),
        StructField("age", FloatType()),
        StructField("hypertension", IntegerType()),
        StructField("heart_disease", IntegerType()),
        StructField("ever_married", StringType()),
        StructField("Residence_type", StringType()),
        StructField("stroke", IntegerType()),
        StructField("work_type", StringType()),
    ]
)

new_df = healthcare_df.filter(F.col("bmi") != "N/A")
new_df = spark.createDataFrame(healthcare_df.rdd, schema=schema)


test_df = healthcare_df.filter(F.col("bmi") == "N/A")

Y_train = new_df.select("bmi")
X_train = new_df.select([x for x in new_df.columns if x != "bmi"])

Y_train.show()
Y_test = test_df.select("bmi")
X_test = test_df.select([x for x in new_df.columns if x != "bmi"])

encoder = OneHotEncoder(
    inputCols=[x for x in new_df.columns if x != "bmi"],
    outputCols=[str(x) + "vec" for x in new_df.columns if x != "bmi"],
)

model = encoder.fit(Y_train.withColumnRenamed("bmi", "id"))
Y_train = model.transform(Y_train)

Y_train.show()

# strokes_df = healthcare_df.filter(F.col("stroke") == 1).cache()
# strokes_df_count = strokes_df.count()

# workTypes_df = (
#     strokes_df.groupBy("work_type")
#     .count()
#     .withColumn("%worktype", F.round(F.col("count") / strokes_df_count * 100, scale=2))
# )

# # Of those who have strokes
# # +-------------+-----+---------+
# # |    work_type|count|%worktype|
# # +-------------+-----+---------+
# # |Self-employed|   65|     26.1|
# # |      Private|  149|    59.84|
# # |     children|    2|      0.8|
# # |     Govt_job|   33|    13.25|
# # +-------------+-----+---------+

# married_df = (
#     strokes_df.groupBy("ever_married")
#     .count()
#     .withColumn("%married", F.round(F.col("count") / strokes_df_count * 100, scale=2))
# )

# # +------------+-----+--------+
# # |ever_married|count|%married|
# # +------------+-----+--------+
# # |          No|   29|   11.65|
# # |         Yes|  220|   88.35|
# # +------------+-----+--------+

# smoking_df = (
#     strokes_df.groupBy("smoking_status")
#     .count()
#     .filter(F.col("smoking_status") != "Unknown")
#     .withColumn("%smoking", F.round(F.col("count") / strokes_df_count * 100, scale=2))
# )

# # +---------------+-----+--------+
# # | smoking_status|count|%smoking|
# # +---------------+-----+--------+
# # |         smokes|   42|   16.87|
# # |   never smoked|   90|   36.14|
# # |formerly smoked|   70|   28.11|
# # +---------------+-----+--------+
