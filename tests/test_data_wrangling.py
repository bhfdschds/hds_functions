"""Unit tests for data_wrangling.py.

These tests validate the correctness of common DataFrame transformations, including:
    - clean_column_names: Sanitizes and deduplicates column names
    - map_column_values: Maps column values based on a provided dictionary

Edge cases tested:
    - Handling of special characters and leading digits in column names
    - Duplicate column names and uniqueness enforcement
    - Nonexistent columns and existing destination columns in mapping
    - Empty mapping dictionaries and null handling in unmapped values

PySpark's assertDataFrameEqual is used to ensure accurate DataFrame comparison.
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.testing import assertDataFrameEqual

from hds_functions.data_wrangling import (
    clean_column_names,
    map_column_values,
)


@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing."""
    return (
        SparkSession.builder.master("local[1]")
        .appName("data-wrangling-tests")
        .getOrCreate()
    )


def test_clean_column_names_basic(spark):
    """Test that special characters and leading digits are cleaned correctly."""
    df = spark.createDataFrame([(1, 2)], ["Col@Name!", "0@ther#Name"])
    result = clean_column_names(df)
    expected = spark.createDataFrame([(1, 2)], ["col_name_", "_0_ther_name"])
    assertDataFrameEqual(result, expected)


def test_clean_column_names_duplicates(spark):
    """Test that duplicate column names are made unique with suffixes."""
    df = spark.createDataFrame([(1, 2, 3)], ["A", "A", "A"])
    result = clean_column_names(df)
    assert result.columns == ["a", "a_2", "a_3"]


def test_map_column_values_overwrite(spark):
    """Test value mapping when overwriting the original column."""
    df = spark.createDataFrame([("A",), ("B",), ("C",)], ["label"])
    map_dict = {"A": "Apple", "B": "Banana"}
    result = map_column_values(df, map_dict, column="label")
    expected = spark.createDataFrame([("Apple",), ("Banana",), (None,)], ["label"])
    assertDataFrameEqual(result, expected)


def test_map_column_values_new_column(spark):
    """Test value mapping into a new column without overwriting the original."""
    df = spark.createDataFrame([("X",), ("Y",)], ["type"])
    map_dict = {"X": "Xylophone"}
    result = map_column_values(df, map_dict, column="type", new_column="mapped")
    expected = spark.createDataFrame(
        [("X", "Xylophone"), ("Y", None)], ["type", "mapped"]
    )
    assertDataFrameEqual(result, expected)


def test_map_column_values_column_not_found(spark):
    """Test that mapping raises an error when the target column does not exist."""
    df = spark.createDataFrame([("foo",)], ["bar"])
    with pytest.raises(ValueError, match="Column 'baz' does not exist"):
        map_column_values(df, {"foo": "bar"}, column="baz")


def test_map_column_values_empty_dict(spark):
    """Test that mapping with an empty dictionary raises a ValueError."""
    df = spark.createDataFrame([("val",)], ["col"])
    with pytest.raises(ValueError, match="Empty mapping dictionary provided"):
        map_column_values(df, {}, column="col")


def test_map_column_values_existing_new_column(spark):
    """Test that mapping into an already existing column name raises an error."""
    df = spark.createDataFrame([("a", "b")], ["col", "mapped"])
    with pytest.raises(ValueError, match="Column 'mapped' already exists"):
        map_column_values(df, {"a": "A"}, column="col", new_column="mapped")
