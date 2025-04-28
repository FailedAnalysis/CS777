import pyspark
import time

from pyspark.sql import DataFrame, SparkSession
from typing import List
import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.metrics import accuracy_score
import numpy as np

spark= SparkSession \
       .builder \
       .appName("Our First Spark Example") \
       .getOrCreate()


df = spark.read.csv('gs://cs777dataset/expression_matrix_labeled.csv', header=True, inferSchema=True)


indexer = StringIndexer(inputCol="label", outputCol="label_index")

gene_cols = [col for col in df.columns if col not in ["label", "cell"]]
assembler = VectorAssembler(inputCols = gene_cols, outputCol="features")

preprocess_pipeline = Pipeline(stages=[indexer, assembler])
preprocessed_model = preprocess_pipeline.fit(df)
processed_df = preprocessed_model.transform(df).select("cell", "features", "label_index")


train_df, test_df = processed_df.randomSplit([0.8, 0.2], seed=42)

start = time.time()

lr = LogisticRegression(featuresCol="features", labelCol="label_index")
lr_model = lr.fit(train_df)
lr_preds = lr_model.transform(test_df)
end = time.time()
print("Logistic run time", end - start)


start = time.time()
rf = RandomForestClassifier(featuresCol="features", labelCol="label_index", numTrees=100)
rf_model = rf.fit(train_df)
rf_preds = rf_model.transform(test_df)
end = time.time()
print("Random Forrest run time", end - start)


start = time.time()
gbt = GBTClassifier(featuresCol="features", labelCol="label_index", maxIter=50)
gbt_model = gbt.fit(train_df)
gbt_preds = gbt_model.transform(test_df)
end = time.time()
print("GBTC run time", end - start)

evaluator = BinaryClassificationEvaluator(labelCol="label_index")

print("Logistic Regression AUC:", evaluator.evaluate(lr_preds))
print("RF AUC:", evaluator.evaluate(rf_preds))
print("GBTC AUC:", evaluator.evaluate(gbt_preds))