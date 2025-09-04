"""Unit tests for data_aggregation.py.

These tests validate the correctness of row-selection utilities including:
    - first_row
    - first_rank
    - first_dense_rank

Edge cases tested:
    - Handling of NULL values
    - Global (unpartitioned) top-N selection
    - Index column inclusion
    - Ties in rank and dense_rank
    - Input validation and error handling

PySpark's assertDataFrameEqual is used to ensure accurate DataFrame comparison.
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.testing import assertDataFrameEqual

from hds_functions.data_aggregation import (
    first_dense_rank,
    first_rank,
    first_row,
    select_top_rows,
)


@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession fixture for running tests."""
    return (
        SparkSession.builder.appName("data-aggregation-tests")
        .master("local[*]")
        .getOrCreate()
    )


@pytest.mark.parametrize("func", [first_row, first_rank, first_dense_rank])
def test_top_1_per_partition(spark, func):
    """Test that top 1 row per group is returned for all methods."""
    df = spark.createDataFrame(
        [
            ("A", 3),
            ("A", 1),
            ("B", 2),
            ("B", 4),
        ],
        ["group", "value"],
    )

    result = func(df, n=1, partition_by=["group"], order_by=[F.asc("value")])
    expected = spark.createDataFrame(
        [
            ("A", 1),
            ("B", 2),
        ],
        ["group", "value"],
    )

    assertDataFrameEqual(result.orderBy("group"), expected.orderBy("group"))


def test_null_values_first_row(spark):
    """Test that NULLs are treated as smallest values by default."""
    df = spark.createDataFrame(
        [
            ("A", None),
            ("A", 2),
            ("A", 1),
        ],
        ["group", "value"],
    )

    result = first_row(df, n=2, partition_by=["group"], order_by=[F.asc("value")])
    expected = spark.createDataFrame(
        [
            ("A", None),
            ("A", 1),
        ],
        ["group", "value"],
    )

    assertDataFrameEqual(result.orderBy("value"), expected.orderBy("value"))


def test_null_values_explicit_ordering(spark):
    """Test NULL ordering behavior using asc_nulls_last()."""
    df = spark.createDataFrame(
        [
            ("A", None),
            ("A", 2),
            ("A", 1),
        ],
        ["group", "value"],
    )

    result = first_row(
        df, n=2, partition_by=["group"], order_by=[F.col("value").asc_nulls_last()]
    )
    expected = spark.createDataFrame(
        [
            ("A", 1),
            ("A", 2),
        ],
        ["group", "value"],
    )

    assertDataFrameEqual(result.orderBy("value"), expected.orderBy("value"))


def test_unpartitioned_input(spark):
    """Test behavior when no partitioning is specified (global top-N)."""
    df = spark.createDataFrame(
        [
            ("A", 3),
            ("B", 1),
            ("C", 2),
        ],
        ["group", "value"],
    )

    result = first_row(df, n=2, partition_by=None, order_by=[F.asc("value")])
    expected = spark.createDataFrame(
        [
            ("B", 1),
            ("C", 2),
        ],
        ["group", "value"],
    )

    assertDataFrameEqual(result.orderBy("value"), expected.orderBy("value"))


def test_rank_with_ties(spark):
    """Test that rank includes all tied rows in the same rank."""
    df = spark.createDataFrame(
        [
            ("A", 1),
            ("A", 1),
            ("A", 2),
        ],
        ["group", "value"],
    )

    result = first_rank(df, n=1, partition_by=["group"], order_by=[F.asc("value")])
    expected = spark.createDataFrame(
        [
            ("A", 1),
            ("A", 1),
        ],
        ["group", "value"],
    )

    assertDataFrameEqual(result.orderBy("value"), expected.orderBy("value"))


def test_dense_rank_multiple_ranks(spark):
    """Test that dense_rank correctly returns rows with top N distinct ranks."""
    df = spark.createDataFrame(
        [
            ("A", 1),
            ("A", 1),
            ("A", 2),
            ("A", 3),
        ],
        ["group", "value"],
    )

    result = first_dense_rank(
        df, n=2, partition_by=["group"], order_by=[F.asc("value")]
    )
    expected = spark.createDataFrame(
        [
            ("A", 1),
            ("A", 1),
            ("A", 2),
        ],
        ["group", "value"],
    )

    assertDataFrameEqual(result.orderBy("value"), expected.orderBy("value"))


def test_index_column_included(spark):
    """Test that index column is returned when return_index_column=True."""
    df = spark.createDataFrame(
        [
            ("A", 2),
            ("A", 1),
        ],
        ["group", "value"],
    )

    result = first_row(
        df,
        n=1,
        partition_by=["group"],
        order_by=[F.asc("value")],
        return_index_column=True,
        index_column_name="custom_index",
    )

    assert "custom_index" in result.columns
    assert result.count() == 1
    row = result.first()
    assert row["custom_index"] == 1


def test_invalid_method_raises(spark):
    """Test that invalid ranking method raises an assertion error."""
    df = spark.createDataFrame([("A", 1)], ["group", "value"])

    with pytest.raises(AssertionError, match="Invalid method"):
        select_top_rows(df, method="invalid", n=1)


def test_dummy_column_collision_raises(spark):
    """Test that a collision on '_dummy_column' raises a ValueError."""
    df = spark.createDataFrame([("A", 1)], ["_dummy_column"])

    with pytest.raises(ValueError, match="already contains '_dummy_column'"):
        select_top_rows(df, method="row_number", n=1, partition_by=None)
