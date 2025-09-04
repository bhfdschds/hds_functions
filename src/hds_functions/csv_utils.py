"""CSV utilities for reading and writing Spark DataFrames.

Functions:
    - read_csv_file: Reads a CSV into a Spark DataFrame.
    - write_csv_file: Writes a Spark DataFrame to a CSV file.
    - create_dict_from_csv: Reads a CSV and creates a dict from specified key and value
        columns.
"""

import os

import pandas as pd
from pyspark.sql import DataFrame

from .environment_utils import get_spark_session, resolve_path


def read_csv_file(
    path: str, repo: str = None, keep_default_na: bool = False, **kwargs
) -> DataFrame:
    """Read a CSV file and return a Spark DataFrame.

    Args:
        path (str): CSV file path (absolute, relative, or repo-relative).
        repo (str, optional): Repo name if path is repo-relative.
        keep_default_na (bool): Whether to include default NaN values.
            Defaults to False.
        **kwargs: Additional args for pd.read_csv().

    Returns:
        DataFrame: Spark DataFrame with the CSV data.

    Example:
        >>> read_csv_file('./relative/path/in/project.csv')
        >>> read_csv_file('/Workspace/absolute/path.csv')
        >>> read_csv_file(path='path/in/repo.csv', repo='common_repo')
    """
    # Resolve file path considering repo context
    resolved_path = resolve_path(path, repo)

    # Read CSV into pandas DataFrame with optional NA handling
    pandas_df = pd.read_csv(resolved_path, keep_default_na=keep_default_na, **kwargs)

    # Get SparkSession and convert pandas DataFrame to Spark DataFrame
    spark = get_spark_session()
    spark_df = spark.createDataFrame(pandas_df)

    return spark_df


def write_csv_file(
    df: DataFrame,
    path: str,
    repo: str = None,
    index: bool = False,
    max_rows_threshold: int = 1000,
    **kwargs,
) -> None:
    """Write a Spark DataFrame to a CSV file.

    Args:
        df (DataFrame): Spark DataFrame to write.
        path (str): CSV file path (absolute, relative, or repo-relative).
        repo (str, optional): Repo name if path is repo-relative.
        index (bool): Include index in CSV. Defaults to False.
        max_rows_threshold (int): Max rows allowed before error. Defaults to 1000.
        **kwargs: Additional args for pd.DataFrame.to_csv().

    Raises:
        ValueError: If DataFrame is empty, too large, or dir missing.
        IOError: If writing the CSV fails.

    Example:
        >>> write_csv_file(spark_df, './relative/path/in/project.csv')
        >>> write_csv_file(spark_df, '/Workspace/absolute/path.csv')
        >>> write_csv_file(spark_df, path='path/in/repo.csv', repo='common_repo')
    """
    # Count rows in the DataFrame
    row_count = df.count()

    # Raise error if DataFrame too large
    if row_count > max_rows_threshold:
        raise ValueError(
            f"DataFrame exceeds maximum rows threshold of {max_rows_threshold}. "
            "This function is for small datasets. Use save_table() for large datasets."
        )

    # Resolve file path, considering repo-relative paths if applicable
    resolved_path = resolve_path(path, repo)

    # Ensure target directory exists before writing
    directory = os.path.dirname(resolved_path)
    if not os.path.exists(directory):
        raise ValueError(f"Directory '{directory}' does not exist.")

    # Raise error if DataFrame is empty (nothing to write)
    if row_count == 0:
        raise ValueError("DataFrame is empty")

    try:
        # Convert Spark DataFrame to Pandas DataFrame and write CSV
        df.toPandas().to_csv(resolved_path, index=index, **kwargs)
    except Exception as err:
        # Wrap and raise IOError on failure to write CSV
        raise IOError("Error writing DataFrame to CSV file") from err


def create_dict_from_csv(
    path: str,
    key_column: str,
    value_columns,
    retain_column_names: bool = False,
    cast_key_as_string: bool = True,
    repo: str = None,
) -> dict:
    """Reads a CSV and creates a dict with keys and values from specified columns.

    Args:
        path (str): CSV file path (absolute, relative, or repo-relative).
        key_column (str): Column used as dict keys.
        value_columns (list or str): Column(s) for dict values.
        retain_column_names (bool, optional): If True, keep column names in values dict.
            Defaults to False.
        cast_key_as_string (bool, optional): If True, cast keys to strings.
            Defaults to True.
        repo (str, optional): Repo name if path is repo-relative. Defaults to None.

    Returns:
        dict: Keys from key_column with values from value_columns.

    Raises:
        ValueError: If key_column contains duplicates.

    Example:
        >>> result = create_dict_from_csv('./data.csv', 'Name', ['Age', 'Gender'],
        ...                              retain_column_names=False)
        >>> print(result)
        {'John': ['30', 'Male'], 'Alice': ['25', 'Female']}
        >>> result = create_dict_from_csv('./data.csv', 'Name', ['Age', 'Gender'],
        ...                              retain_column_names=True)
        >>> print(result)
        {'John': {'Age': '30', 'Gender': 'Male'}, 'Alice': {'Age': '25', 'Female'}}
    """
    # Ensure value_columns is a list, even if single column provided as string
    if isinstance(value_columns, str):
        value_columns = [value_columns]

    # Resolve full file path, handling repo-relative if specified
    resolved_path = resolve_path(path, repo)

    # Read CSV file into pandas DataFrame
    df = pd.read_csv(resolved_path)

    # Check that the key column has unique values (no duplicates allowed)
    if not df[key_column].is_unique:
        raise ValueError("Key column '{}' is not unique".format(key_column))

    result_dict = {}
    # Iterate through each row in DataFrame
    for _, row in df.iterrows():
        # Cast key to string if requested
        key = str(row[key_column]) if cast_key_as_string else row[key_column]

        # Extract the values for the specified columns
        values = {col: row[col] for col in value_columns}

        # Simplify values if only one column specified
        if len(value_columns) == 1:
            values = next(iter(values.values()))
        # Convert to list if not retaining column names in values dict
        elif not retain_column_names:
            values = list(values.values())

        # Assign processed values to corresponding key in result dictionary
        result_dict[key] = values

    return result_dict
