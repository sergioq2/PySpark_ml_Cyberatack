import os
import json
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler

spark = SparkSession.builder.appName('Session')\
    .config('spark.master', 'local[*]')\
    .config('spark.executor.memory', '8g')\
    .config('spark.driver.memory', '8g')\
    .config('spark.shuffle.sql.partitions', 100)\
    .getOrCreate()

def read_parquet(path):
    df = spark.read.parquet(path)
    return df

def features_columns(df):
    feature_cols = df.columns
    feature_cols.remove('target')
    feature_cols.remove('__index_level_0__')
    for col in df.columns:
        if '.' in col:
            feature_cols.remove(col)
    return feature_cols

def feature_engineerin(df, feature_cols):
    indexer = StringIndexer(inputCol="target", outputCol="label").fit(df)
    df = indexer.transform(df)
    vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = vector_assembler.transform(df)
    #scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    # df = scaler.fit(df).transform(df)
    df = df.select("label", "features")
    return df

def train_test_split(df):
    train, test = df.randomSplit([0.7, 0.3], seed=12345)
    return train, test

def model_train(train):
    num_partitions = 100
    train = train.repartition(num_partitions)
    rf = RandomForestClassifier(featuresCol="features", labelCol="label")
    param_grid = (ParamGridBuilder()
                  .addGrid(rf.numTrees, [20, 50, 100])
                  .addGrid(rf.maxDepth, [5, 10, 15])
                  .build())
    cross_validator = CrossValidator(estimator=rf,
                                     estimatorParamMaps=param_grid,
                                     evaluator=MulticlassClassificationEvaluator(),
                                     numFolds=5)
    cv_model = cross_validator.fit(train)
    try:
        cv_model.save("../../model_trained")
    except Exception as e:
        print("Error saving the model: ", str(e))
    return cv_model

def accuracy_score(test, model):
    predictions = model.transform(test)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    return accuracy

def main():
    df = read_parquet("../../parquet_data/iot_dataset.parquet")
    feature_cols = features_columns(df)
    df = feature_engineerin(df, feature_cols)
    train, test = train_test_split(df)
    model = model_train(train)
    accuracy = accuracy_score(test, model)
    return accuracy

if __name__ == "__main__":
    main()