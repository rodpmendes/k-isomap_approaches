from datetime import datetime

print("Data e hora formatada:", ('table_results_'+ datetime.now().strftime("%Y%m%d_%H%M%S") +'.csv'))


from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from pyspark.sql.functions import monotonically_increasing_id

# Criando uma sessão Spark
spark = SparkSession.builder \
    .appName("Exemplo de DataFrame Spark") \
    .getOrCreate()

# Definindo o esquema para o DataFrame
schema = StructType([
    StructField("Id", IntegerType(), True),
    StructField("dataset_name", StringType(), True),
    StructField("classes", StringType(), True),
    StructField("samples_per_class", StringType(), True),
    StructField("samples_total", IntegerType(), True),
    StructField("dimensionality", IntegerType(), True),
    StructField("dr_name", StringType(), True),
    StructField("dr_learning", StringType(), True),
    StructField("dr_transformation", StringType(), True),
    StructField("dr_structure", StringType(), True),
    StructField("classifier_name", StringType(), True),
    StructField("classifier_accuracy", DoubleType(), True),
    StructField("classifier_kappa", DoubleType(), True),
    StructField("classifier_balanced_accuracy", DoubleType(), True)
])

# Dados exemplo
data = [
    (1, 'Dataset1', 'ClassA, ClassB', '10, 20', 30, 100, 'DR1', 'Learning1', 'Transformation1', 'Structure1', 'Classifier1', 0.85, 0.75, 0.80),
    (2, 'Dataset2', 'ClassC, ClassD', '30, 40', 70, 200, 'DR2', 'Learning2', 'Transformation2', 'Structure2', 'Classifier2', 0.75, 0.65, 0.70),
    (3, 'Dataset3', 'ClassE, ClassF', '50, 60', 110, 300, 'DR3', 'Learning3', 'Transformation3', 'Structure3', 'Classifier3', 0.90, 0.80, 0.85)
]

# Criando o DataFrame Spark
df_spark = spark.createDataFrame(data, schema)

# Adicionando três novas linhas ao DataFrame
novos_dados = [
    (4, 'Dataset4', 'ClassG, ClassH', '70, 80', 150, 400, 'DR4', 'Learning4', 'Transformation4', 'Structure4', 'Classifier4', 0.95, 0.85, 0.90),
    (5, 'Dataset5', 'ClassI, ClassJ', '90, 100', 190, 500, 'DR5', 'Learning5', 'Transformation5', 'Structure5', 'Classifier5', 0.85, 0.75, 0.80),
    (6, 'Dataset6', 'ClassK, ClassL', '110, 120', 230, 600, 'DR6', 'Learning6', 'Transformation6', 'Structure6', 'Classifier6', 0.90, 0.80, 0.85)
]

# Adicionando as novas linhas ao DataFrame Spark
#df_spark = df_spark.union(spark.createDataFrame(novos_dados, schema))

# Adicionando uma coluna de ID incremental
#df_spark = df_spark.withColumn("Id", monotonically_increasing_id()+1)

# Exibindo o DataFrame Spark
df_spark.show()

# Encerrando a sessão Spark
spark.stop()