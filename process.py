import findspark
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, GBTClassifier, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


def count_with_condition(condition):
    """Utility function to count only specific rows based on the 'condition'."""
    return F.count(F.when(condition, True))


def count_distinct_with_condition(condition, values):
    """Utility function to count only distinct & specific rows based on the 'condition'."""
    return F.count_distinct(F.when(condition, values))


def run(pipeline, paramGrid, train_df, test_df):
    """
    Main function used to train & test a given model.
    The training step uses cross-validation to find the best hyper-parameters for the model.

    :param pipeline: Model pipeline.
    :param paramGrid: Parameter grid used for cross-validation.
    :param train_df: Training dataframe.
    :param test_df: Testing dataframe.
    :return: the best model from cross-validation
    """

    fitted_model = fit_model(paramGrid, pipeline, train_df)
    evaluate_model(fitted_model, test_df)

    return fitted_model


def fit_model(paramGrid, pipeline, train_df):
    """
    Function that trains the model using cross-validation.
    Also, it prints the best validation results and hyper-parameters.

    :param paramGrid: Parameter grid used for cross-validation.
    :param pipeline: Model pipeline.
    :param train_df: Training dataframe.
    :return: the best model from cross-validation
    """

    crossval = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=MulticlassClassificationEvaluator(metricName="f1", beta=1.0),
        parallelism=3,
        numFolds=3
    )

    fitted_model = crossval.fit(train_df)
    print_best_validation_score(fitted_model)
    print_best_parameters(fitted_model)

    return fitted_model


def create_pipeline(model):
    """
    Create a pipeline based on a model.

    :param model: The end model that will be used for training.
    :return: the built pipeline.
    """

    scaler = StandardScaler(inputCol="unscaled_features", outputCol="features")
    pipeline = Pipeline(stages=[scaler, model])

    return pipeline


def print_best_validation_score(cross_validation_model):
    """Prints the best validation score based on the results from the cross-validation model."""
    print()
    print("-" * 60)
    print(f"F1 score, on the validation split, for the best model: {np.max(cross_validation_model.avgMetrics) * 100:.2f}%")
    print("-" * 60)


def print_best_parameters(cross_validation_model):
    """Prints the best hyper-parameters based on the results from the cross-validation model."""

    parameters = cross_validation_model.getEstimatorParamMaps()[np.argmax(cross_validation_model.avgMetrics)]

    print()
    print("-" * 60)
    print("Best model hyper-parameters:")
    for param, value in parameters.items():
        print(f"{param}: {value}")
    print("-" * 60)


def evaluate_model(model, test_df):
    """Evaluate the model on the test set using F1 score and print the results."""

    predictions = model.transform(test_df)
    evaluator =  MulticlassClassificationEvaluator(metricName="f1", beta=1.0)
    metric = evaluator.evaluate(predictions)

    print()
    print("-" * 60)
    print(f"F1 score, on the test set is: {metric*100:.2f}%")
    print("-" * 60)

    return metric


if __name__ == "__main__":
    findspark.init()

    print("Creating spark session...")
    spark = SparkSession \
        .builder \
        .appName("Sparkify Churn Prediction") \
        .master("local[*]") \
        .config("spark.ui.port", "4041") \
        .getOrCreate()

    print("Loading data...")
    EVENT_DATA_LINK = "mini_sparkify_event_data.json"
    df = spark.read.json(EVENT_DATA_LINK)
    df.persist()

    print("Cleaning data...")
    # Drop unregistered users.
    cleaned_df = df.filter(F.col("userId") != "")
    # Fill the length of the song with 0.
    # Fill the artist and the song with a string constant to signal that those pages don't have such information.
    cleaned_df = cleaned_df.fillna({
        "length": 0,
        "artist": "unknown",
        "song": "unknown"
    })

    print("Defining churn label...")
    labeled_df = cleaned_df.withColumn(
        "churnEvent",
        F.when(F.col("page") == "Cancellation Confirmation", 1).otherwise(0)
    )
    labeled_df = labeled_df.withColumn("churn", F.sum("churnEvent").over(Window.partitionBy("userId")))
    labeled_df = labeled_df.withColumn("churn", F.when(F.col("churn") >= 1, 1).otherwise(0))

    print("Creating features...")
    marked_df = labeled_df.withColumn("listeningToMusic", F.when(F.col("page") == "NextSong", 1).otherwise(0))
    aggregated_df = labeled_df.groupby("userId").agg(
        F.count("page").alias("numPagesVisited"),
        count_with_condition(F.col("page") == "NextSong").alias("numTotalPlays"),
        count_distinct_with_condition(F.col("artist") != "unknown", F.col("artist")).alias("numTotalArtists"),
        F.max("churn").alias("churn")
    )
    assembler = VectorAssembler(
        inputCols=["numPagesVisited", "numTotalPlays", "numTotalArtists"],
        outputCol="unscaled_features"
    )
    engineered_df = assembler.transform(aggregated_df)
    engineered_df = engineered_df.select(F.col("unscaled_features"), F.col("churn").alias("label"))

    print("Splitting the data...")
    train_df, test_df = engineered_df.randomSplit([0.8, 0.2], seed=42)

    print("Train classifier...")
    lr = LogisticRegression()
    pipeline = create_pipeline(lr)
    # paramGrid = ParamGridBuilder() \
    #     .addGrid(lr.maxIter, [10, 25, 50]) \
    #     .addGrid(lr.regParam, [0.05, 0.1, 0.2]) \
    #     .addGrid(lr.elasticNetParam, [0.05, 0.1, 0.2]) \
    #     .build()
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.maxIter, [10]) \
        .addGrid(lr.regParam, [0.05]) \
        .addGrid(lr.elasticNetParam, [0.05]) \
        .build()

    fitted_model = run(pipeline, paramGrid, train_df.alias("train_df_lr"), test_df.alias("test_df_lr"))

    print("Save best fitted model...")
    fitted_model.write().save("classifier")

    print("Stopping spark session...")
    spark.stop()
