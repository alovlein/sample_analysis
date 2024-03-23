import aws_config
import pyspark
import pyspark.sql.functions as F
import delta
import pandas as pd
import plotly.express as px
import scipy


def spark_init():
    aws_access_key = aws_config.aws_access_key
    aws_secret_key = aws_config.aws_secret_key

    builder = pyspark.sql.SparkSession.builder.appName("MyApp") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.hadoop.fs.s3a.access.key", aws_access_key) \
        .config("spark.hadoop.fs.s3a.secret.key", aws_secret_key) \
        .config("spark.databricks.delta.retentionDurationCheck.enabled", "false")
    extra_packages = ["org.apache.hadoop:hadoop-aws:3.3.1"]
    spark = delta.configure_spark_with_delta_pip(builder, extra_packages=extra_packages).getOrCreate()
    return spark


def load_data(mode: str = 'csv') -> pd.DataFrame:
    """
    Load data for the interview technical screen
    :param mode: str input is either s3 or csv, just to show we know how to load csv in a pinch
    :return: pandas dataframe with the data provided
    """

    assert mode in ('s3', 'csv'), 'Mode can be either s3 or csv'

    if mode == 's3':
        # there are many requirements to running this locally
        # but you cant anyway without my personal AWS credentials
        # using spark and S3 is way overkill for 17 records, but good practice for the big data you probably have
        ## install java via brew install java then simlink into /Library ...
        ## install spark via https://archive.apache.org/dist/spark/ (ensure version matches AWS Glue)
        ## install pyspark via pip install pyspark and pip install delta-spark

        spark = spark_init()
        sdf = spark.read.format('delta').load('s3a://lovlein-tech-demos/teiko/lake/cell_count/')
        df = sdf.toPandas()
    elif mode == 'csv':
        df = pd.read_csv('cell-count.csv')

    return df


def relative_frequency(populations: list[str]) -> pd.DataFrame:
    # you asked for python code, so lets ditch pyspark / sql and go with traditional pandas
    # the commented out spark code is what runs on the site
    #spark = spark_init()
    #df = spark.read.format('delta').load('s3a://lovlein-tech-demos/teiko/lake/cell_count/').toPandas()
    df = load_data('csv')
    # you want percentage of each population, but that will be easier after the melt (unpivot)
    df_p = pd.melt(df, id_vars=['sample'], value_vars=populations, var_name='population', value_name='count')
    df_p['total_count'] = df_p['count'].groupby(df_p['sample']).transform('sum')
    df_p['percentage'] = (100 * df_p['count'] / df_p['total_count']).round(2)
    return df_p.sort_values(by='percentage', ascending=False)


def response_comparison(population: str, populations: list[str]):
    # we can reuse some of the stuff above. I could do this more efficiently but with only 4 lines lets just copy/paste
    #spark = spark_init()
    #sdf = spark.read.format('delta').load('s3a://lovlein-tech-demos/teiko/lake/cell_count/')
    #df_reduced = sdf\
    #    .filter(F.col('treatment') == 'tr1')\
    #    .filter(F.col('condition') == 'melanoma')\
    #    .filter(F.col('sample_type') == 'PBMC')\
    #    .toPandas()
    df = load_data('csv')
    df_reduced = df[df['treatment'] == 'tr1']
    df_reduced = df_reduced[df_reduced['condition'] == 'melanoma']
    df_reduced = df_reduced[df_reduced['sample_type'] == 'PBMC']
    df_p = pd.melt(
        df_reduced,
        id_vars=['sample', 'response'],
        value_vars=populations,
        var_name='population',
        value_name='count'
    )
    df_p['total_count'] = df_p['count'].groupby(df_p['sample']).transform('sum')
    df_p['percentage'] = (100 * df_p['count'] / df_p['total_count']).round(2)

    df_reduced = df_p[df_p['population'] == population]

    pos_response = df_reduced[df_reduced['response'] == 'y']['percentage'].to_numpy()
    neg_response =df_reduced[df_reduced['response'] == 'n']['percentage'].to_numpy()

    t_res = scipy.stats.ttest_ind(pos_response, neg_response, equal_var=False)
    p_value = round(t_res.pvalue, 2)
    significance_string = 'potentially significantly different'
    if p_value > 0.05:
        significance_string = 'probably not significantly different'

    fig = px.box(
        df_reduced,
        x="response",
        y="percentage",
        points='all',
        title=f'{population} response is {significance_string} with p-value {p_value}',
        labels={
            "percentage": "Relative frequency [%]",
            "response": "Response"
        },
    )
    return fig


def database_queries(query: str) -> pd.DataFrame:
    # I might be reading into this, but I assume here you want to see SQL.
    # I prefer to use pyspark, but efficiency wise there is no difference and SQL may be easier to read in this case
    # These all require spark and may not run on your machine without extensive set up
    available_queries = ('interview_3', 'interview_4', 'interview_5a', 'interview_5b', 'interview_5c')
    assert query in available_queries, f'Please request a pre-written query from: {available_queries}'
    spark = spark_init()
    sdf = spark.read.format('delta').load('s3a://lovlein-tech-demos/teiko/lake/cell_count/')
    sdf.createOrReplaceTempView('cell_count')

    match query:
        case 'interview_3':
            query = '''
            select condition, count(distinct subject) as number_of_subjects
            from cell_count
            group by condition
            '''
        case 'interview_4':
            query = '''
            select sample
            from cell_count
            where sample_type = 'PBMC' and time_from_treatment_start = 0 and treatment = 'tr1'
            '''
        case 'interview_5a':
            query = '''
            select project, count(distinct sample) as number_of_samples
            from cell_count
            where sample_type = 'PBMC' and time_from_treatment_start = 0 and treatment = 'tr1'
            group by project
            order by project asc
            '''
        case 'interview_5b':
            query = '''
            select response, count(*) as number_of_responses
            from cell_count
            where sample_type = 'PBMC' and time_from_treatment_start = 0 and treatment = 'tr1'
            group by response
            order by response asc
            '''
        case 'interview_5c':
            query = '''
            select sex, count(*) as number_of_subjects
            from cell_count
            where sample_type = 'PBMC' and time_from_treatment_start = 0 and treatment = 'tr1'
            group by sex
            order by sex asc
            '''

    sql_df = spark.sql(query)
    return sql_df.toPandas()
