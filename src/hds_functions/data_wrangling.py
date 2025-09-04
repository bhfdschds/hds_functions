"""Module for common data wrangling tasks in PySpark.

Functions:
    - clean_column_names: Cleans column names of a PySpark DataFrame.
    - map_column_values: Maps column values using a dictionary.
"""

from itertools import chain
from typing import Dict

import pyspark.sql.functions as F
from pyspark.sql import DataFrame


def clean_column_names(df: DataFrame) -> DataFrame:
    """Clean column names by replacing invalid characters and ensuring uniqueness.

    Replaces non-alphanumeric characters with underscores, ensures names don't
    start with a number, and converts them to lowercase. Duplicate column names
    are made unique by appending a numeric suffix.

    Args:
        df (DataFrame): The DataFrame whose column names are to be cleaned.

    Returns:
        DataFrame: A new DataFrame with cleaned column names.

    Example:
        >>> data = [("John Doe", 30), ("Jane Smith", 25)]
        >>> df = spark.createDataFrame(data, ["Name", "Age"])
        >>> df = df.select("Name", F.col("Name").alias("0_N@me!"),
        ...                F.col("Name").alias("0_N@me!"))
        >>> df_cleaned = clean_column_names(df)
        >>> df_cleaned.columns
        ['name', '_0_n_me_', '_0_n_me__2']
    """

    def clean_name(name: str) -> str:
        # Replace non-alphanumeric characters with underscores
        cleaned_name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
        # Ensure column name doesn't start with a number
        if cleaned_name[0].isdigit():
            cleaned_name = "_" + cleaned_name
        return cleaned_name.lower()

    # Clean column names
    cleaned_columns = [clean_name(col) for col in df.columns]

    # Check for duplicate column names and make them unique
    seen = {}
    new_columns = []
    for col in cleaned_columns:
        if col not in seen:
            seen[col] = 1
            new_columns.append(col)
        else:
            seen[col] += 1
            new_columns.append(f"{col}_{seen[col]}")

    # Rename columns in the DataFrame
    return df.toDF(*new_columns)


def map_column_values(
    df: DataFrame, map_dict: Dict, column: str, new_column: str = ""
) -> DataFrame:
    """Map column values using a dictionary.

    Maps values in the specified column to new values defined in `map_dict`.
    If `new_column` is specified, results are stored there; otherwise, the
    original column is overwritten.

    Args:
        df (DataFrame): DataFrame to operate on.
        map_dict (Dict): Dictionary containing source and target values.
        column (str): Name of the column to map.
        new_column (str, optional): Name of the new column to store mapped values.
            Defaults to overwriting `column`.

    Returns:
        DataFrame: DataFrame with values mapped as specified.

    Example:
        >>> data = [('A',), ('B',), ('C',), ('D',)]
        >>> df = spark.createDataFrame(data, ['column_to_map'])
        >>> map_dict = {'A': 'Apple', 'B': 'Banana', 'C': 'Cherry'}
        >>> mapped_df = map_column_values(
        ...     df, map_dict, column='column_to_map', new_column='mapped_column'
        ... )
        >>> mapped_df.show()
        +--------------+--------------+
        |column_to_map |mapped_column |
        +--------------+--------------+
        |      A       |     Apple    |
        |      B       |    Banana    |
        |      C       |    Cherry    |
        |      D       |      null    |
        +--------------+--------------+
    """
    # Check that the column to map exists in the DataFrame
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    # Raise an error if the mapping dictionary is empty
    if not map_dict:
        raise ValueError("Empty mapping dictionary provided.")

    # Convert the Python dictionary into a Spark map expression
    # e.g., {'A': 'Apple'} -> create_map(lit('A'), lit('Apple'))
    spark_map = F.create_map(*[F.lit(x) for x in chain(*map_dict.items())])

    # If a new column name is given, make sure it doesn't already exist
    if new_column and new_column in df.columns:
        raise ValueError(f"Column '{new_column}' already exists in the DataFrame.")

    # Use new_column if provided, otherwise overwrite the original column
    new_col_name = new_column or column

    # Create the new column by applying the Spark map to the original column
    return df.withColumn(new_col_name, spark_map[df[column]])
