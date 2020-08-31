from pyspark.sql import SparkSession,functions

if __name__ == "__main__":
	#create sparkSession
    sc= SparkSession.builder.getOrCreate()

    # create a DataFrame from a csv file
    df = sc.read.csv(header=True, path='hdfs://devenv/user/spark/instacart/data/reorder_counts.csv')

    #change schema
    for col in df.columns:
        df = df.withColumn(col,functions.col(col).cast("int"))

    #print the schema
    #df.printSchema()

    #write to hdfs in parquet file format
    df.write.parquet('hdfs://devenv/user/spark/instacart/data/reorderCounts_in_parquet')