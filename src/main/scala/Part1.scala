  /* Author: TULIKA MITHAL
      */

  import org.apache.log4j.{Level, Logger}
  import org.apache.spark.sql.SparkSession
  import org.apache.spark.ml.feature.StopWordsRemover
  import org.apache.spark.ml.Pipeline
  import org.apache.spark.ml.classification.LogisticRegression
  import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
  import org.apache.spark.ml.feature.StringIndexer
  import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
  import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
  import org.apache.spark.ml.classification.NaiveBayes
  import org.apache.spark.sql.functions.{col, monotonically_increasing_id}


  object Part1 {

    def main(args: Array[String]): Unit = {

      val spark = SparkSession.builder.appName("Simple Application").getOrCreate()

      Logger.getLogger("GraphFrames").setLevel(Level.ALL)
      spark.sparkContext.setLogLevel("ERROR")
      // in Scala

      //Reading file data
      val tweetData = spark.read.option("header","true").option("inferSchema","true")
        .csv(args(0))


      val relevantTweetData = tweetData.filter(col("text") isNotNull)
      val indexedData = relevantTweetData.withColumn("id",monotonically_increasing_id())


      val Array(trainingData, testData) = indexedData.randomSplit(Array(0.8, 0.2))

      val tokenizer = new Tokenizer()
        .setInputCol("text")
        .setOutputCol("words")

      val remover = new StopWordsRemover()
        .setInputCol("words")
        .setOutputCol("filtered")

      val hashingTF = new HashingTF()
        .setNumFeatures(1000)
        .setInputCol(remover.getOutputCol)
        .setOutputCol("features")

      val indexer = new StringIndexer()
        .setInputCol("airline_sentiment")
        .setOutputCol("label")

      val pipeline = new Pipeline()
        .setStages(Array(tokenizer, remover, hashingTF, indexer))

      val model = pipeline.fit(trainingData)

      val preprocessedData = model.transform(trainingData).select("features","label","id")

      val lr = new LogisticRegression()
        .setMaxIter(10)
        .setRegParam(0.3)
        .setElasticNetParam(0.8)

      val paramGrid = new ParamGridBuilder()
        .addGrid(lr.regParam, Array(0.1, 0.01))
        .build()


      val cv_lr = new CrossValidator()
        .setEstimator(lr)
        .setEvaluator(new BinaryClassificationEvaluator)
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(2)



      val cvModel_lr = cv_lr.fit(preprocessedData)



      val pipeline1 = new Pipeline()
        .setStages(Array(tokenizer, remover, hashingTF,indexer))

      val  test_model = pipeline1.fit(testData)

      val preprocessedTestData = test_model.transform(testData).select("features","id", "label")

      //val writer = new PrintWriter(new File(args(1)))

      val res = cvModel_lr.transform(preprocessedTestData)
        .select("label", "prediction")


      import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
      val evaluator = new MulticlassClassificationEvaluator()
      evaluator.setLabelCol("label")
      evaluator.setMetricName("accuracy")
      val accuracy = evaluator.evaluate(res)

      println(accuracy)

    }

  }


