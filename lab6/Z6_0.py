from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator

SEED = 0

def main():
    spark = SparkSession.builder.appName("-").getOrCreate()

    df = spark.read.csv("datasets/mammographic.csv", inferSchema=True)

    assembler = VectorAssembler(inputCols=["_c0","_c1","_c2","_c3","_c4"], outputCol="features")
    df = assembler.transform(df).withColumnRenamed("_c5", "label")

    df.show()
    train_data, test_data = df.randomSplit([0.5, 0.5], seed=SEED)

    lr = LogisticRegression(featuresCol="features", labelCol="label")
    param_grid = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1]).build()

    split = TrainValidationSplit(estimator=lr,
                                 estimatorParamMaps=param_grid,
                                 evaluator=BinaryClassificationEvaluator(labelCol="label"),
                                 trainRatio=0.8)

    model = split.fit(train_data).bestModel

    print(f"Najlepszy parametr regularyzacji={model.getRegParam()}")

    predictions = model.transform(test_data)
    predictions.select("features", "probability", "prediction", "label").show(20, truncate=False)

if __name__=="__main__":
    main()
