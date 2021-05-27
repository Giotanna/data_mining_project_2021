from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import FloatType, StructField, StructType, IntegerType, StringType, NullType
from pyspark.ml.regression import LinearRegression

spark = SparkSession.builder.getOrCreate()

schema = StructType(
    [
        StructField("id", IntegerType()),
        StructField("gender", StringType()),
        StructField("age", FloatType()),
        StructField("hypertension", IntegerType()),
        StructField("heart_disease", IntegerType()),
        StructField("ever_married", StringType()),
        StructField("work_type", StringType()),
        StructField("Residence_type", StringType()),
        StructField("avg_glucose_level", FloatType()),
        StructField("bmi", FloatType()),
        StructField("smoking_status", StringType()),
        StructField("stroke", IntegerType()),
    ]
)
healthcare_df = (
    spark.read.option("delimiter", ",").csv("healthcare-dataset-stroke-data.csv", header=True, schema=schema)
    # .filter(F.col("gender") != "Other")
)


assembler = VectorAssembler(
    inputCols=["age", "hypertension", "heart_disease", "avg_glucose_level", "stroke"], outputCol="features"
)

# Make vectors out of numerical values

# Dataframe without null values
train_df = assembler.transform(healthcare_df.filter(healthcare_df.bmi.isNotNull()))

# Dataframe with null values
test_df = assembler.transform(healthcare_df.filter(healthcare_df.bmi.isNull()))

# Initialize model
lr = LinearRegression(labelCol="bmi")

# Fit the model based on the dataframe that does not contain null values
lr_model = lr.fit(train_df)

unlabeled_data = test_df.select("features")

# Make a 'prediction' dataframe based on the vectors assembled
prediction = lr_model.transform(unlabeled_data)

# Fill the missing values in the dataframe that contains null values.
test_df = (
    test_df.drop("bmi")
    .join(prediction, on=test_df.features == prediction.features, how="inner")
    .drop("features")
    .withColumn("bmi", F.round(F.col("prediction"), scale=1))
    .drop("prediction")
    .select(
        "id",
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "Residence_type",
        "avg_glucose_level",
        "bmi",
        "smoking_status",
        "stroke",
    )
)

# Update the original dataframe
healthcare_df = test_df.union(train_df.drop("features"))
healthcare_df = healthcare_df.drop("smoking_status")
healthcare_df.show()
