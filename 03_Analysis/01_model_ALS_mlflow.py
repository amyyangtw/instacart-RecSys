from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
import mlflow
import mlflow.spark
import os

if __name__ == "__main__":

	os.environ["GIT_PYTHON_REFRESH"] = "quiet"
	#create sparkSession
	spark= SparkSession.builder.getOrCreate()

	print('loading data...')
	# Load Data
	data = spark.read.parquet('hdfs://devenv/user/spark/instacart/data/reorderCounts_in_parquet_v2')
	# Splite Data
	def randomSplitByUser(df, weights, seed=None):
	    trainingRation = weights[0]
	    fractions = {row['user_id']: trainingRation for row in df.select('user_id').distinct().collect()}
	    training = df.sampleBy('user_id', fractions, seed)
	    testRDD = df.rdd.subtract(training.rdd)
	    test = spark.createDataFrame(testRDD, df.schema)
	    return (training, test)
	(train_data, test_data) = randomSplitByUser(data, weights=[0.7, 0.3])
	nbTraining = train_data.count()
	nbTesting = test_data.count()
	print("Training: %d, test: %d" %(nbTraining, nbTesting))
	train_data = train_data.cache()
	test_data = test_data.cache()

	print('seting ml pipeline')
	rank=8
	maxIter=10
	regParam=0.03
	alpha = 1
	implicitPrefs = 'True'
	mlflow.log_param("rank", rank)
	mlflow.log_param("maxIter", maxIter)
	mlflow.log_param("regParam", regParam)
	mlflow.log_param("alpha", alpha)
	mlflow.log_param("implicitPrefs", implicitPrefs)

	als = ALS(
		maxIter=maxIter, 
		rank=rank, 
		regParam=regParam,
		alpha = alpha, 
		implicitPrefs=True,
		userCol="user_id", itemCol="product_id", ratingCol="count",
		coldStartStrategy="drop")
	# Define evaluator as RMSE
	evaluator = RegressionEvaluator(metricName="rmse",labelCol="count",predictionCol="prediction")

	print('training model...')
	tunedModel = als.fit(train_data)
	print('testing model ....')
	predictions = tunedModel.transform(test_data)
	test_metric = evaluator.evaluate(predictions)
	mlflow.log_metric('test_' + evaluator.getMetricName(), test_metric)

	print('get all predictions....')
	all_predictions = bestModel.transform(data)
	all_predictions.write.format('csv').option('header', True).mode('overwrite').option('sep', ',').save('./result_tmp/all_predictions.csv')

	# Generate top 10 item recommendations for each user
	userRecs = model.recommendForAllUsers(10)
	
	#save ranking result to csv
	result_df = userRecs.toPandas()
	userIds = [int(ids) for ids in result_df['user_id']]
	ranking = ['rank_' + str(i) for i in range(1, 11)]
	df = pd.DataFrame(np.zeros((206080, 10), dtype=np.int),index=userIds, columns=ranking)
	df.index.name = 'user_id'
	df.reset_index(level=0, inplace=True)

	for ids in userIds:
    user_id = int(ids)
    rank10 = []
    for rec in result_df[result_df['user_id'] == user_id]['recommendations']:
        for row in rec:
            product_ids = re.findall('product_id=\d+', str(row))
            for ids in product_ids:
                num = int(ids.replace('product_id=', ''))
                rank10.append(num)      
    for i in range(0,10):
        value = rank10[i]
        i+=1
        col = 'rank_' + str(i)
        df.loc[df['user_id'] == user_id, col] = value
	df.to_csv('./result_tmp/model_ALS_rank.csv', index = 0)
    

