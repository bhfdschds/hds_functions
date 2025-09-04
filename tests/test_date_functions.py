"""Unit tests for date_functions.py.

These tests validate the correctness of date parsing and conversion utilities,
including:
    - validate_date_string: Checks if a date string is valid
    - parse_date_instruction: Converts date strings into Spark SQL expressions
    - convert_date_units_to_days: Converts units like days, weeks, months, or years
      into days

Edge cases tested:
    - Leap year and non-leap year date validation
    - Invalid or nonsensical date strings
    - Handling of None or empty input
    - Complex date arithmetic (adding/subtracting days, weeks, months, or years)
    - Invalid units in date expressions

PySpark's assertDataFrameEqual is used to verify that the generated DataFrames
match expected outputs.
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, to_date
from pyspark.sql.types import DateType, StructField, StructType
from pyspark.testing import assertDataFrameEqual

from hds_functions.date_functions import (
    convert_date_units_to_days,
    parse_date_instruction,
    validate_date_string,
)


@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing."""
    return (
        SparkSession.builder.master("local[1]")
        .appName("data-wrangling-tests")
        .getOrCreate()
    )


@pytest.mark.parametrize(
    "date_str,expected",
    [
        ("2020-01-01", True),  # Valid normal date
        ("2020-02-30", False),  # Invalid date (Feb 30 does not exist)
        ("2019-02-28", True),  # Valid non-leap year Feb end date
        ("2019-02-29", False),  # Invalid non-leap year Feb 29
        ("2020-02-29", True),  # Valid leap year Feb 29
        ("", False),  # Empty string is invalid
        ("2020-13-01", False),  # Invalid month (13 does not exist)
        ("2020-00-10", False),  # Invalid month zero
        ("2020-01-00", False),  # Invalid day zero
        ("not-a-date", False),  # Non-date string
    ],
)
def test_validate_date_string(date_str, expected):
    """Test validation of date strings for correctness."""
    assert validate_date_string(date_str) is expected


@pytest.mark.parametrize(
    "input_str,expected_output",
    [
        ("2020-01-01", "date('2020-01-01')"),
        ("index_date + 5 days", "index_date + cast(round(5*1) as int)"),
        ("x - 6 weeks", "x - cast(round(6*7) as int)"),
        ("index_date + 3 months", "index_date + cast(round(3*30) as int)"),
        ("index_date - 2 years", "index_date - cast(round(2*365.25) as int)"),
        ("index_date", "index_date"),
        ("current_date() + 5 days", "current_date() + cast(round(5*1) as int)"),
        (None, "cast(NULL as date)"),
        ("random_expression", "random_expression"),
    ],
)
def test_parse_date_instruction(input_str, expected_output):
    """Test parsing of date instruction strings into expressions."""
    assert parse_date_instruction(input_str) == expected_output


def test_parse_date_instruction_invalid_date():
    """Test that invalid date strings raise ValueError in parse_date_instruction."""
    with pytest.raises(ValueError, match="Invalid date: 2020-02-30"):
        parse_date_instruction("2020-02-30")


@pytest.mark.parametrize(
    "input_expr,expected_expr",
    [
        ("index_date + 6 months", "index_date + cast(round(6*30) as int)"),
        ("x - 7.5 weeks", "x - cast(round(7.5*7) as int)"),
        (
            "index_date - 2 years, x - 7.5 weeks",
            "index_date - cast(round(2*365.25) as int), x - cast(round(7.5*7) as int)",
        ),
        ("date_col + 1 day", "date_col + cast(round(1*1) as int)"),
    ],
)
def test_convert_date_units_to_days(input_expr, expected_expr):
    """Test conversion of date units (days, weeks, months, years) into days."""
    assert convert_date_units_to_days(input_expr) == expected_expr


def test_convert_date_units_to_days_invalid_unit():
    """Test that invalid date units raise ValueError in convert_date_units_to_days."""
    with pytest.raises(ValueError, match="Invalid unit"):
        convert_date_units_to_days("index_date + 5 decades")


@pytest.mark.parametrize(
    "input_str,expected_date",
    [
        ("2020-01-01", "2020-01-01"),
        ("index_date + 5 days", "2020-01-06"),
        ("index_date - 6 weeks", "2019-11-20"),
        ("index_date + 3 months", "2020-03-31"),
        ("index_date - 2 years", "2017-12-31"),
        ("index_date", "2020-01-01"),
        (None, None),
    ],
)
def test_parse_date_instruction_creates_correct_dates(spark, input_str, expected_date):
    """Test that parse_date_instruction generates correct Spark SQL date expressions."""
    df = spark.createDataFrame(
        [("2020-01-01",)],
        ["index_date"],
    ).withColumn("index_date", to_date("index_date"))

    expr_str = parse_date_instruction(input_str)

    result_df = df.select(expr(expr_str).alias("result"))

    if expected_date is None:
        expected_df = spark.createDataFrame(
            [(None,)], schema=StructType([StructField("result", DateType(), True)])
        )
    else:
        expected_df = spark.createDataFrame([(expected_date,)], ["result"]).withColumn(
            "result", to_date("result")
        )

    assertDataFrameEqual(result_df, expected_df)
