/* Author: TULIKA MITHAL
*/


import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.fpm.FPGrowth



object Part3 {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder.appName("Simple Application").getOrCreate()

    val transaction_data = spark.read.option("header","true")
      .csv(args(0)).select("order_id","product_id")

    //val product_data = spark.read.option("header","true")
    //  .csv(args(1)).select("product_id","product_name")

    //val data = transaction_data.join(product_data,transaction_data.col("product_id") === product_data.col("product_id")

    val data = transaction_data.groupBy("order_id").agg(collect_list("product_id")).withColumnRenamed("collect_list(product_id)","items").select("items")

    val fpgrowth = new FPGrowth().setItemsCol("items").setMinSupport(0.1).setMinConfidence(0.1)
    val model = fpgrowth.fit(data)

    //val writer = new PrintWriter(new File(args(1)))

    // Display frequent itemsets.
    println(model.freqItemsets)

    // Display generated association rules.
    println(model.associationRules)

    model.transform(data).show()
  }

}


