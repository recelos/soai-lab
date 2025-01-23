from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

SEED = 0

def main():
    spark = SparkSession.builder.appName("-").getOrCreate()

    df = spark.read.csv("datasets/mammographic.csv", inferSchema=True)

    assembler = VectorAssembler(inputCols=["_c0","_c1","_c2","_c3","_c4"], outputCol="features")
    df = assembler.transform(df).withColumnRenamed("_c5", "label")

    df.show()
    train_data, test_data = df.randomSplit([0.5, 0.5], seed=SEED)


    dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")
    dt_param_grid = ParamGridBuilder().addGrid(dt.maxDepth, [5, 10, 15]).addGrid(dt.minInstancesPerNode, [10,20,30]).build()

    lr = LogisticRegression(featuresCol="features", labelCol="label")
    lr_param_grid = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1, 1]).addGrid(lr.maxIter, [10, 20, 30]).build()

    dt_cv = CrossValidator(estimator=dt,
                           estimatorParamMaps=dt_param_grid,
                           evaluator=BinaryClassificationEvaluator(labelCol="label"),
                           numFolds=5,
                           seed=SEED)

    lr_cv = CrossValidator(estimator=lr,
                           estimatorParamMaps=lr_param_grid,
                           evaluator=BinaryClassificationEvaluator(labelCol="label"),
                           numFolds=5,
                           seed=SEED)


    lr_model = lr_cv.fit(train_data)
    dt_model = dt_cv.fit(train_data)

    lr_best_model = lr_model.bestModel
    dt_best_model = dt_model.bestModel

    lr_best_metric = max(lr_model.avgMetrics)
    dt_best_metric = max(dt_model.avgMetrics)

    print(f"Średnia wartość metryki walidacyjnej dla DecisionTree: {dt_best_metric}")
    print(f"Parametry najlepszego modelu DecisionTree: maxDepth={dt_best_model.getMaxDepth()}, minInstancesPerNode={dt_best_model.getMinInstancesPerNode()}")
    print()
    print(f"Średnia wartość metryki walidacyjnej dla LogisticRegression: {lr_best_metric}")
    print(f"Parametry najlepszego modelu LogisticRegression: regParam={lr_best_model.getRegParam()}, maxIter={lr_best_model.getMaxIter()}")
    
    
    dt_predictions = dt_best_model.transform(test_data)
    dt_predictions.select("features", "probability", "prediction", "label").show(20, truncate=False)

    lr_predictions = lr_best_model.transform(test_data)
    lr_predictions.select("features", "probability", "prediction", "label").show(20, truncate=False)

if __name__=="__main__":
    main()
