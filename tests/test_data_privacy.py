"""Unit tests for data_privacy.py.

These tests validate the correctness of functions for rounding and redacting
sensitive numeric data in Spark DataFrames, including:
    - round_counts_to_multiple: Rounds numeric columns to a specified multiple
    - redact_low_counts: Redacts (masks) counts below a specified threshold

Edge cases tested:
    - Invalid input types and missing columns
    - Multiple columns rounding and redaction
    - Custom redaction values (None, strings, integers)
    - Integration of rounding followed by redaction using DataFrame.transform()

PySpark's assertDataFrameEqual is used to ensure accurate DataFrame comparison.
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.testing import assertDataFrameEqual

from hds_functions.data_privacy import (
    redact_low_counts,
    round_counts_to_multiple,
)


@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing."""
    return (
        SparkSession.builder.master("local[1]")
        .appName("data-privacy-tests")
        .getOrCreate()
    )


def test_round_counts_to_multiple_basic(spark):
    """Test rounding counts to nearest multiple of 5 on single column."""
    data = [(1, 7), (2, 17), (3, 22)]
    df = spark.createDataFrame(data, ["id", "count"])
    result = round_counts_to_multiple(df, ["count"], multiple=5)

    expected_data = [(1, 5), (2, 15), (3, 20)]
    expected_df = spark.createDataFrame(expected_data, ["id", "count"])

    assertDataFrameEqual(result, expected_df)


def test_round_counts_to_multiple_multiple_columns(spark):
    """Test rounding counts to nearest multiple on multiple columns."""
    data = [(1, 7, 12), (2, 17, 25)]
    df = spark.createDataFrame(data, ["id", "count1", "count2"])
    result = round_counts_to_multiple(df, ["count1", "count2"], multiple=10)

    expected_data = [(1, 10, 10), (2, 20, 30)]
    expected_df = spark.createDataFrame(expected_data, ["id", "count1", "count2"])

    assertDataFrameEqual(result, expected_df)


def test_round_counts_to_multiple_errors(spark):
    """Test error handling for invalid inputs to round_counts_to_multiple."""
    df = spark.createDataFrame([(1, 7)], ["id", "count"])

    with pytest.raises(TypeError):
        round_counts_to_multiple("not a df", ["count"])

    with pytest.raises(TypeError):
        round_counts_to_multiple(df, "count")  # columns not a list

    with pytest.raises(TypeError):
        round_counts_to_multiple(df, [1])  # columns list not strings

    with pytest.raises(ValueError):
        round_counts_to_multiple(df, ["missing_col"])  # col doesn't exist

    with pytest.raises(ValueError):
        round_counts_to_multiple(df, ["count"], multiple=0)

    with pytest.raises(ValueError):
        round_counts_to_multiple(df, ["count"], multiple=-5)


def test_redact_low_counts_basic(spark):
    """Test basic redaction of counts below threshold with None as redaction."""
    data = [(1, 7), (2, 17), (3, 3)]
    df = spark.createDataFrame(data, ["id", "count"])
    result = redact_low_counts(df, ["count"], threshold=10)

    expected_data = [(1, None), (2, 17), (3, None)]
    expected_df = spark.createDataFrame(expected_data, ["id", "count"])

    assertDataFrameEqual(result, expected_df)


def test_redact_low_counts_with_redaction_value_string(spark):
    """Test redaction of counts below threshold using a string redaction value."""
    data = [(1, 7), (2, 17)]
    df = spark.createDataFrame(data, ["id", "count"])
    result = redact_low_counts(df, ["count"], threshold=10, redaction_value="REDACTED")

    expected_data = [(1, "REDACTED"), (2, 17)]
    expected_df = spark.createDataFrame(expected_data, ["id", "count"])

    assertDataFrameEqual(result, expected_df)


def test_redact_low_counts_multiple_columns(spark):
    """Test redaction on multiple columns with custom redaction value."""
    data = [(1, 7, 15), (2, 17, 4)]
    df = spark.createDataFrame(data, ["id", "count1", "count2"])
    result = redact_low_counts(
        df, ["count1", "count2"], threshold=10, redaction_value="X"
    )

    expected_data = [(1, "X", 15), (2, 17, "X")]
    expected_df = spark.createDataFrame(expected_data, ["id", "count1", "count2"])

    assertDataFrameEqual(result, expected_df)


def test_redact_low_counts_errors(spark):
    """Test error handling for invalid inputs to redact_low_counts."""
    df = spark.createDataFrame([(1, 7)], ["id", "count"])

    with pytest.raises(ValueError):
        redact_low_counts(df, ["count"], threshold=0)

    with pytest.raises(ValueError):
        redact_low_counts(df, ["count"], threshold=-10)

    with pytest.raises(TypeError):
        redact_low_counts(df, "count", threshold=5)  # columns not a list

    with pytest.raises(TypeError):
        redact_low_counts(df, [1], threshold=5)  # columns not strings

    with pytest.raises(ValueError):
        redact_low_counts(df, ["missing_col"], threshold=5)  # col missing


def test_round_and_redact_integration(spark):
    """Integration test chaining rounding and redaction with .transform()."""
    data = [(1, 7), (2, 17), (3, 22)]
    df = spark.createDataFrame(data, ["id", "count"])

    result = df.transform(
        lambda d: round_counts_to_multiple(d, columns=["count"], multiple=5)
    ).transform(
        lambda d: redact_low_counts(
            d, columns=["count"], threshold=10, redaction_value=0
        )
    )

    expected_data = [(1, 0), (2, 15), (3, 20)]
    expected_df = spark.createDataFrame(expected_data, ["id", "count"])

    assertDataFrameEqual(result, expected_df)
