"""Utilities for redacting and rounding sensitive data in Spark DataFrames.

Functions:
    - round_counts_to_multiple: Round specified numeric columns to a given multiple.
    - redact_low_counts: Redact values in columns below a given threshold.
"""

from typing import List, Optional, Union

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import LongType


def round_counts_to_multiple(
    df: DataFrame, columns: List[str], multiple: int = 5
) -> DataFrame:
    """Round specified numeric columns to the nearest given multiple.

    Args:
        df (DataFrame): Input Spark DataFrame containing columns to round.
        columns (List[str]): List of column names to be rounded.
        multiple (int): Multiple to round values to. Defaults to 5.

    Returns:
        DataFrame: New Spark DataFrame with specified columns rounded.

    Raises:
        TypeError: If df is not a DataFrame, columns is not a list of strings, or
            multiple is not an int.
        ValueError: If columns do not exist or multiple is not positive.

    Example:
        >>> df = spark.createDataFrame([(1, 7), (2, 17)], ["id", "count"])
        >>> rounded_df = round_counts_to_multiple(df, ["count"], multiple=5)
        >>> rounded_df.show()
        +---+-----+
        | id|count|
        +---+-----+
        |  1|    5|
        |  2|   20|
        +---+-----+
    """
    # Validate that input is a Spark DataFrame
    if not isinstance(df, DataFrame):
        raise TypeError("The input 'df' must be a Spark DataFrame.")

    # Ensure 'columns' is a list of strings
    if not isinstance(columns, list) or not all(
        isinstance(col, str) for col in columns
    ):
        raise TypeError("The 'columns' argument must be a list of strings.")

    # Ensure 'multiple' is a positive integer
    if not isinstance(multiple, int) or multiple <= 0:
        raise ValueError("The 'multiple' argument must be a positive integer.")

    # Verify that all specified columns exist in the DataFrame
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"The column '{col}' does not exist in the DataFrame.")

    # Round each specified column to the nearest multiple
    for col in columns:
        df = df.withColumn(
            col, (F.round(F.col(col) / multiple) * multiple).cast(LongType())
        )

    return df


def redact_low_counts(
    df: DataFrame,
    columns: List[str],
    threshold: int,
    redaction_value: Optional[Union[str, int]] = None,
) -> DataFrame:
    """Redact values in columns below a given threshold.

    Masks counts in specified DataFrame columns below a threshold.

    Args:
        df (DataFrame): Spark DataFrame containing the data.
        columns (List[str]): Column names where counts will be redacted.
        threshold (int): Threshold below which counts are redacted. Must be positive.
        redaction_value (Optional[Union[str, int]]): Value replacing redacted counts.
            If None, redacted counts set to None.
            If string, column cast to string type.

    Returns:
        DataFrame: New DataFrame with counts below threshold redacted.

    Raises:
        ValueError: If any column in `columns` does not exist or if threshold
            is not positive.
        TypeError: If `threshold` is not int or `columns` not list of strings.

    Example:
        >>> df = spark.createDataFrame([(1, 7), (2, 17)], ["id", "count"])
        >>> redacted_df = redact_low_counts(df, ["count"], threshold=10,
        ...     redaction_value="[:REDACTED:]")
        >>> redacted_df.show()
        +---+------------+
        | id|       count|
        +---+------------+
        |  1|[:REDACTED:]|
        |  2|          17|
        +---+------------+
    """
    # Ensure threshold is a positive integer
    if not isinstance(threshold, int) or threshold <= 0:
        raise ValueError("Threshold must be a positive integer.")

    # Validate that columns is a list of strings
    if not isinstance(columns, list) or not all(
        isinstance(col, str) for col in columns
    ):
        raise TypeError("Columns must be a list of strings.")

    # Check that all specified columns exist in the DataFrame
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")

    # Convert redaction value to a Spark literal for use in expressions
    redaction_value = (
        F.lit(redaction_value) if redaction_value is not None else F.lit(None)
    )

    # Apply redaction: replace values below threshold with redaction_value
    for col in columns:
        df = df.withColumn(
            col, F.when(F.col(col) >= threshold, F.col(col)).otherwise(redaction_value)
        )

    return df
