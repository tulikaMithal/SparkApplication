/* Author: TULIKA MITHAL
*/

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.clustering.DistributedLDAModel


object Part2 {

  def main(args: Array[String]): Unit = {


    if (args.length == 0) {
      println("i need two parameters ")
    }

    val spark = SparkSession.builder.appName("Simple Application").getOrCreate()

    //Reading file data
    val tweetData = spark.read.option("header","true")
      .csv(args(0)).filter(col("text") isNotNull)


    //adding id column
    val indexedData = tweetData.withColumn("id",monotonically_increasing_id())

    //selecting required columns
    val data = indexedData.select("id","airline_sentiment","airline","text","tweet_id")

    //converting airline_sentiment according to given scheme
    val updated_v1 = data.withColumn("airline_sentiment", regexp_replace(col("airline_sentiment"), "negative", "1.0"))
    val updated_v2 = updated_v1.withColumn("airline_sentiment", regexp_replace(col("airline_sentiment"), "positive", "5.0"))
    val updated_v3 = updated_v2.withColumn("airline_sentiment", regexp_replace(col("airline_sentiment"),  "neutral", "2.5"))

    //casting airline_sentiment to int and renaming column as airline_sentiment_index
    val dfNew = updated_v3.withColumn("airline_sentiment_index", updated_v3.col("airline_sentiment").cast("Int"))
      .drop("airline_sentiment").withColumnRenamed("airline_sentiment", "airline_sentiment_index")

   //finding top and worst airline
    val top = dfNew.groupBy("airline").avg("airline_sentiment_index").orderBy(desc("avg(airline_sentiment_index)"))
    val worst = dfNew.groupBy("airline").avg("airline_sentiment_index").orderBy(asc("avg(airline_sentiment_index)"))

    var output = ""

    val top_airline = top.collectAsList().get(0).getString(0)
    val worst_airline = worst.collectAsList().get(0).getString(0)

    output += "Best Airline: " + top_airline +"\n"
    output += "Worst Airline: " + worst_airline +"\n"




    //getting top and worst airline data
    val top_airline_data = dfNew.filter(col("airline") === top_airline)
    val worst_airline_data =dfNew.filter(col("airline") === worst_airline)

    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")

    val remover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filtered")

    val vectorizer = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("features")
      .setVocabSize(2048)


    val lda = new LDA()
      .setK(10)
      .setMaxIter(50)
      .setOptimizer("em")

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, vectorizer, lda))

    val model_top= pipeline.fit(top_airline_data)
    val model_worst = pipeline.fit(worst_airline_data)

    val vectorizerModel_top = model_top.stages(2).asInstanceOf[CountVectorizerModel]

    val ldaModel_top = model_top.stages(3).asInstanceOf[DistributedLDAModel]

    val vocabList_top = vectorizerModel_top.vocabulary
    val termsIdx2Str_top = udf { (termIndices: Seq[Int]) => termIndices.map(idx => vocabList_top(idx)) }

    // Review Results of LDA model with Online Variational Bayes
    val topics = ldaModel_top.describeTopics(maxTermsPerTopic = 5)
      .withColumn("terms", termsIdx2Str_top(col("termIndices")))


    //topics.select("topic", "terms", "termWeights").write.csv(args(1)+"topics_for_top_airline.csv")
   // topics.select("topic", "terms", "termWeights").write.format("txt").save(args(1)+"output.txt")
    //topics.select("topic", "terms", "termWeights").toJSON.write.save(args(1)+"/topairline")
    val row = topics.select("topic", "terms", "termWeights").collectAsList()

    output += "\n"+ row.toArray.mkString(" ")



    val vectorizerModel_worst = model_worst.stages(2).asInstanceOf[CountVectorizerModel]

    val ldaModel_worst = model_worst.stages(3).asInstanceOf[DistributedLDAModel]

    val vocabList_worst = vectorizerModel_worst.vocabulary
    val termsIdx2Str_worst = udf { (termIndices: Seq[Int]) => termIndices.map(idx => vocabList_worst(idx)) }

    // Review Results of LDA model with Online Variational Bayes
    val topics1 = ldaModel_worst.describeTopics(maxTermsPerTopic = 5)
      .withColumn("terms", termsIdx2Str_worst(col("termIndices")))

   // topics1.select("topic", "terms", "termWeights").write.csv(args(1)+"topics_for_worst_airline.csv")
  //  topics1.select("topic", "terms", "termWeights")..save(args(1)+"output.txt")
    //topics1.select("topic", "terms", "termWeights").toJSON.write.save(args(1)+"/worstairline")
    //output += "\n"+topics1.select("topic", "terms", "termWeights").collectAsList().toString

    val row1= topics1.select("topic", "terms", "termWeights").collectAsList()
    output += "\n"+ row1.toArray.mkString(" ")

    val sc = spark.sparkContext
    sc.parallelize(List(output)).saveAsTextFile(args(1))
    sc.stop()
  }

}


