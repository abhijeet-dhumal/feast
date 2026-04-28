from datetime import timedelta
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from pyspark.sql import DataFrame
from tqdm import tqdm

from feast import BatchFeatureView, Field
from feast.aggregation import Aggregation
from feast.infra.common.materialization_job import (
    MaterializationJobStatus,
    MaterializationTask,
)
from feast.infra.common.retrieval_task import HistoricalRetrievalTask
from feast.infra.compute_engines.spark.compute import SparkComputeEngine
from feast.infra.compute_engines.spark.job import SparkDAGRetrievalJob
from feast.infra.compute_engines.spark.utils import (
    _ensure_s3a_event_log_dir,
    get_or_create_new_spark_session,
    map_in_pandas,
    write_to_online_store,
)
from feast.infra.offline_stores.contrib.spark_offline_store.spark import (
    SparkOfflineStore,
)
from feast.types import Float32, Int32, Int64
from tests.component.spark.utils import (
    _check_offline_features,
    _check_online_features,
    create_entity_df,
    create_feature_dataset,
    create_spark_environment,
    driver,
    now,
)


@pytest.mark.integration
def test_spark_compute_engine_get_historical_features():
    spark_environment = create_spark_environment()
    fs = spark_environment.feature_store
    registry = fs.registry
    data_source = create_feature_dataset(spark_environment)

    def transform_feature(df: DataFrame) -> DataFrame:
        df = df.withColumn("conv_rate", df["conv_rate"] * 2)
        df = df.withColumn("acc_rate", df["acc_rate"] * 2)
        return df

    driver_stats_fv = BatchFeatureView(
        name="driver_hourly_stats",
        entities=[driver],
        mode="python",
        aggregations=[
            Aggregation(column="conv_rate", function="sum"),
            Aggregation(column="acc_rate", function="avg"),
        ],
        udf=transform_feature,
        udf_string="transform_feature",
        ttl=timedelta(days=3),
        schema=[
            Field(name="conv_rate", dtype=Float32),
            Field(name="acc_rate", dtype=Float32),
            Field(name="avg_daily_trips", dtype=Int64),
            Field(name="driver_id", dtype=Int32),
        ],
        online=False,
        offline=False,
        source=data_source,
    )

    entity_df = create_entity_df()

    try:
        fs.apply([driver, driver_stats_fv])

        # 🛠 Build retrieval task
        task = HistoricalRetrievalTask(
            project=spark_environment.project,
            entity_df=entity_df,
            feature_view=driver_stats_fv,
            full_feature_name=False,
            registry=registry,
        )

        # 🧪 Run SparkComputeEngine
        engine = SparkComputeEngine(
            repo_config=spark_environment.config,
            offline_store=SparkOfflineStore(),
            online_store=MagicMock(),
        )

        spark_dag_retrieval_job = engine.get_historical_features(registry, task)
        spark_df = cast(SparkDAGRetrievalJob, spark_dag_retrieval_job).to_spark_df()
        df_out = spark_df.orderBy("driver_id").toPandas()

        # ✅ Assert output
        assert df_out.driver_id.to_list() == [1001, 1002]
        assert abs(df_out["sum_conv_rate"].to_list()[0] - 3.1) < 1e-6
        assert abs(df_out["sum_conv_rate"].to_list()[1] - 2.0) < 1e-6
        assert abs(df_out["avg_acc_rate"].to_list()[0] - 1.4) < 1e-6
        assert abs(df_out["avg_acc_rate"].to_list()[1] - 1.0) < 1e-6

    finally:
        spark_environment.teardown()


@pytest.mark.integration
def test_spark_compute_engine_materialize():
    """
    Test the SparkComputeEngine materialize method.
    For the current feature view driver_hourly_stats, The below execution plan:
    1. feature data from create_feature_dataset
    2. filter by start_time and end_time, that is, the last 2 days
        for the driver_id 1001, the data left is row 0
        for the driver_id 1002, the data left is row 2
    3. apply the transform_feature function to the data
        for all features, the value is multiplied by 2
    4. write the data to the online store and offline store

    Returns:

    """
    spark_environment = create_spark_environment()
    fs = spark_environment.feature_store
    registry = fs.registry

    data_source = create_feature_dataset(spark_environment)

    def transform_feature(df: DataFrame) -> DataFrame:
        df = df.withColumn("conv_rate", df["conv_rate"] * 2)
        df = df.withColumn("acc_rate", df["acc_rate"] * 2)
        return df

    driver_stats_fv = BatchFeatureView(
        name="driver_hourly_stats",
        entities=[driver],
        mode="python",
        udf=transform_feature,
        udf_string="transform_feature",
        schema=[
            Field(name="conv_rate", dtype=Float32),
            Field(name="acc_rate", dtype=Float32),
            Field(name="avg_daily_trips", dtype=Int64),
            Field(name="driver_id", dtype=Int32),
        ],
        online=True,
        offline=True,
        source=data_source,
    )

    def tqdm_builder(length):
        return tqdm(total=length, ncols=100)

    try:
        fs.apply([driver, driver_stats_fv])

        # 🛠 Build retrieval task
        task = MaterializationTask(
            project=spark_environment.project,
            feature_view=driver_stats_fv,
            start_time=now - timedelta(days=2),
            end_time=now,
            tqdm_builder=tqdm_builder,
        )

        # 🧪 Run SparkComputeEngine
        engine = SparkComputeEngine(
            repo_config=spark_environment.config,
            offline_store=SparkOfflineStore(),
            online_store=MagicMock(),
            registry=registry,
        )

        spark_materialize_jobs = engine.materialize(registry, task)

        assert len(spark_materialize_jobs) == 1

        assert spark_materialize_jobs[0].status() == MaterializationJobStatus.SUCCEEDED

        _check_online_features(
            fs=fs,
            driver_id=1001,
            feature="driver_hourly_stats:conv_rate",
            expected_value=1.6,
            full_feature_names=True,
        )

        entity_df = create_entity_df()

        _check_offline_features(
            fs=fs,
            feature="driver_hourly_stats:conv_rate",
            entity_df=entity_df,
        )
    finally:
        spark_environment.teardown()


# ---------------------------------------------------------------------------
# Unit tests — no cluster required
# ---------------------------------------------------------------------------


def _base_conf(event_log_dir: str) -> dict:
    return {
        "spark.eventLog.enabled": "true",
        "spark.eventLog.dir": event_log_dir,
        "spark.hadoop.fs.s3a.endpoint": "http://minio:9000",
    }


@patch("feast.infra.compute_engines.spark.utils.boto3")
def test_ensure_s3a_event_log_dir_creates_placeholder_when_empty(mock_boto3):
    """S3A prefix doesn't exist → placeholder object is written."""
    s3 = MagicMock()
    mock_boto3.client.return_value = s3
    s3.list_objects_v2.return_value = {"KeyCount": 0}

    _ensure_s3a_event_log_dir(_base_conf("s3a://my-bucket/spark-events/"))

    s3.list_objects_v2.assert_called_once_with(
        Bucket="my-bucket", Prefix="spark-events/", MaxKeys=1
    )
    s3.put_object.assert_called_once_with(
        Bucket="my-bucket", Key="spark-events/.keep", Body=b""
    )


@patch("feast.infra.compute_engines.spark.utils.boto3")
def test_ensure_s3a_event_log_dir_skips_when_prefix_exists(mock_boto3):
    """S3A prefix already has objects → no placeholder written."""
    s3 = MagicMock()
    mock_boto3.client.return_value = s3
    s3.list_objects_v2.return_value = {"KeyCount": 3}

    _ensure_s3a_event_log_dir(_base_conf("s3a://my-bucket/spark-events/"))

    s3.put_object.assert_not_called()


@patch("feast.infra.compute_engines.spark.utils.boto3")
def test_ensure_s3a_event_log_dir_noop_when_event_log_disabled(mock_boto3):
    """spark.eventLog.enabled != true → boto3 never called."""
    _ensure_s3a_event_log_dir(
        {"spark.eventLog.enabled": "false", "spark.eventLog.dir": "s3a://b/p/"}
    )
    mock_boto3.client.assert_not_called()


@patch("feast.infra.compute_engines.spark.utils.boto3")
def test_ensure_s3a_event_log_dir_noop_for_non_s3a_path(mock_boto3):
    """Non-S3A paths (hdfs://, file://, etc.) are left untouched."""
    _ensure_s3a_event_log_dir(
        {"spark.eventLog.enabled": "true", "spark.eventLog.dir": "hdfs:///spark-logs"}
    )
    mock_boto3.client.assert_not_called()


@patch("feast.infra.compute_engines.spark.utils.boto3")
def test_ensure_s3a_event_log_dir_non_fatal_on_s3_error(mock_boto3):
    """boto3 errors are swallowed — SparkContext will surface its own error."""
    s3 = MagicMock()
    mock_boto3.client.return_value = s3
    s3.list_objects_v2.side_effect = Exception("connection refused")

    _ensure_s3a_event_log_dir(_base_conf("s3a://my-bucket/spark-events/"))


def test_get_or_create_new_spark_session_applies_sql_configs_to_reused_session():
    """SQL/Hadoop configs must be forwarded even when a SparkSession already exists.

    SparkSession.getOrCreate() silently drops new spark_config overrides when an
    active session is found.  get_or_create_new_spark_session() must apply
    spark.sql.* and spark.hadoop.* settings via session.conf.set() afterward.
    """
    mock_session = MagicMock()
    spark_config = {
        "spark.sql.sources.useV1SourceList": "avro",
        "spark.hadoop.fs.s3a.endpoint": "http://minio:9000",
        "spark.executor.instances": "2",  # SparkContext-level key — must NOT be set
    }

    with patch(
        "feast.infra.compute_engines.spark.utils.SparkSession"
    ) as mock_spark_cls:
        mock_spark_cls.getActiveSession.return_value = mock_session

        result = get_or_create_new_spark_session(spark_config)

    assert result is mock_session
    set_calls = {call.args[0] for call in mock_session.conf.set.call_args_list}
    assert "spark.sql.sources.useV1SourceList" in set_calls
    assert "spark.hadoop.fs.s3a.endpoint" in set_calls
    assert "spark.executor.instances" not in set_calls


def test_map_in_pandas_dummy_yield_has_correct_schema():
    """map_in_pandas must yield a DataFrame with column 'status' (int), not
    integer-indexed column '0'.  The wrong column name causes
    ArrowStreamPandasUDFSerializer._create_array to raise
    AttributeError: 'list' object has no attribute 'dtype'."""
    import pandas as pd

    batches = list(map_in_pandas(iter([]), MagicMock()))
    assert len(batches) == 1, "Expected exactly one sentinel batch"
    df = batches[0]
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["status"], (
        f"Expected column ['status'], got {list(df.columns)}"
    )
    assert df["status"].iloc[0] == 0


def test_write_to_online_store_skips_empty_partitions():
    """write_to_online_store must not call online_write_batch for empty partitions."""
    from pyspark.sql import SparkSession
    from pyspark.sql.types import FloatType, StringType, StructField, StructType

    spark = SparkSession.builder.master("local[1]").appName("test_write").getOrCreate()

    schema = StructType(
        [
            StructField("review_id", StringType(), True),
            StructField("val", FloatType(), True),
        ]
    )
    df = spark.createDataFrame([], schema=schema)

    mock_artifacts = MagicMock()
    mock_online_store = MagicMock()
    mock_fv = MagicMock()
    mock_fv.entity_columns = []
    mock_config = MagicMock()
    mock_config.materialization_config.online_write_batch_size = None
    mock_artifacts.unserialize.return_value = (
        mock_fv,
        mock_online_store,
        MagicMock(),
        mock_config,
    )

    write_to_online_store(df, mock_artifacts)

    mock_online_store.online_write_batch.assert_not_called()
    spark.stop()


def test_write_to_online_store_calls_write_batch_for_nonempty_partition():
    """write_to_online_store must call online_write_batch once per non-empty partition."""
    from pyspark.sql import SparkSession
    from pyspark.sql.types import FloatType, StringType, StructField, StructType

    spark = SparkSession.builder.master("local[1]").appName("test_write2").getOrCreate()

    schema = StructType(
        [
            StructField("review_id", StringType(), True),
            StructField("val", FloatType(), True),
        ]
    )
    df = spark.createDataFrame([("r1", 1.0), ("r2", 2.0)], schema=schema).repartition(1)

    mock_artifacts = MagicMock()
    mock_online_store = MagicMock()
    mock_fv = MagicMock()
    mock_fv.entity_columns = []
    mock_fv.features = []
    mock_fv.batch_source.timestamp_field = "val"
    mock_fv.batch_source.created_timestamp_column = None
    mock_config = MagicMock()
    mock_config.materialization_config.online_write_batch_size = None
    mock_config.entity_key_serialization_version = 3
    mock_artifacts.unserialize.return_value = (
        mock_fv,
        mock_online_store,
        MagicMock(),
        mock_config,
    )

    with patch(
        "feast.infra.compute_engines.spark.utils._convert_arrow_to_proto",
        return_value=[],
    ):
        write_to_online_store(df, mock_artifacts)

    mock_online_store.online_write_batch.assert_called_once()
    spark.stop()


def test_spark_embed_model_singleton_per_executor():
    """spark_embed() must load SentenceTransformer at most once per (model, device) pair.

    Verifies:
    1. Output DataFrame has an 'embedding' ArrayType(FloatType()) column.
    2. SentenceTransformer is constructed exactly once (module-level cache).
    3. All rows carry an embedding value.
    """
    import numpy as np
    from pyspark.sql import SparkSession
    from pyspark.sql.types import ArrayType, FloatType, StringType, StructField, StructType

    from feast.infra.compute_engines.spark import utils as spark_utils

    spark_utils._FEAST_EMBED_MODEL_CACHE.clear()

    spark = SparkSession.builder.master("local[1]").appName("test_embed").getOrCreate()

    schema = StructType([StructField("text", StringType(), True)])
    df = spark.createDataFrame(
        [("hello world",), ("feast rocks",), ("embeddings are cool",)], schema=schema
    ).repartition(1)

    fake_model = MagicMock()
    fake_model.encode.return_value = np.zeros((3, 4), dtype="float32")

    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model) as mock_cls:
        with patch("torch.cuda.is_available", return_value=False):
            result = spark_utils.spark_embed(df, text_col="text", model="test-model")
            rows = result.collect()

    assert "embedding" in result.columns
    emb_field = result.schema["embedding"]
    assert isinstance(emb_field.dataType, ArrayType)
    assert isinstance(emb_field.dataType.elementType, FloatType)
    assert len(rows) == 3
    for row in rows:
        assert row.embedding is not None
    # Model constructed exactly once (singleton cache)
    mock_cls.assert_called_once_with("test-model", device="cpu")
    # persist().count() materialised the data; original RDD lineage is gone
    assert result.is_cached

    spark.stop()
    spark_utils._FEAST_EMBED_MODEL_CACHE.clear()


def test_spark_embed_appends_custom_output_col():
    """spark_embed() with a custom output_col produces that column."""
    import numpy as np
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StringType, StructField, StructType

    from feast.infra.compute_engines.spark import utils as spark_utils

    spark_utils._FEAST_EMBED_MODEL_CACHE.clear()

    spark = SparkSession.builder.master("local[1]").appName("test_embed_col").getOrCreate()

    schema = StructType([StructField("body", StringType(), True)])
    df = spark.createDataFrame([("foo",), ("bar",)], schema=schema).repartition(1)

    fake_model = MagicMock()
    fake_model.encode.return_value = np.zeros((2, 8), dtype="float32")

    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model):
        with patch("torch.cuda.is_available", return_value=False):
            result = spark_utils.spark_embed(df, text_col="body", output_col="vec")

    assert "vec" in result.columns
    assert "body" in result.columns

    spark.stop()
    spark_utils._FEAST_EMBED_MODEL_CACHE.clear()


if __name__ == "__main__":
    test_spark_compute_engine_get_historical_features()
