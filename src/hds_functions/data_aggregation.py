"""Aggregation utilities for PySpark DataFrames.

Functions:
    - select_top_rows: Wrapper for first_dense_rank(), first_rank(), and first_row().
    - first_row: Returns the first N rows per partition by sort order.
    - first_rank: Returns rows with the top N ranks per partition.
    - first_dense_rank: Returns rows with the top N dense ranks per partition.

"""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def select_top_rows(
    df,
    method,
    n=1,
    partition_by=None,
    order_by=None,
    return_index_column=False,
    index_column_name="row_index",
) -> DataFrame:
    """Select top N rows per partition using a row indexing method.

    Args:
        df (DataFrame): PySpark DataFrame to process.
        method (str): Row indexing method: 'row_number', 'rank', or 'dense_rank'.
        n (int, optional): Number of rows to retain per partition. Defaults to 1.
        partition_by (list, optional): Columns to partition by. Defaults to None.
        order_by (list, optional): Columns to order within partitions. Defaults to None.
        return_index_column (bool, optional): Include the index column in output.
            Defaults to False.
        index_column_name (str, optional): Name of the index column.
            Defaults to 'row_index'.

    Note:
        PySpark treats nulls as smallest by default in ordering. To customize null
        ordering, use `asc_nulls_last()`, `desc_nulls_first()`, or similar functions.

    Returns:
        DataFrame: PySpark DataFrame with top N rows per partition.

    Examples:
        >>> df = spark.createDataFrame([
        ...     ("A", 1), ("A", 1), ("A", 2),
        ...     ("A", 3), ("B", 4), ("B", 5), ("B", 6)
        ... ], ["group", "value"])
        >>> select_top_rows(df, method="row_number", n=2, partition_by=["group"],
        ...                 order_by=["value"], return_index_column=True,
        ...                 index_column_name="row_number").show()
        >>> select_top_rows(df, method="rank", n=2, partition_by=["group"],
        ...                 order_by=["value"], return_index_column=True,
        ...                 index_column_name="rank_index").show()
        >>> select_top_rows(df, method="dense_rank", n=2, partition_by=["group"],
        ...                 order_by=["value"], return_index_column=True,
        ...                 index_column_name="dense_rank_index").show()
    """
    # Input validation for method
    assert method in ["row_number", "rank", "dense_rank"], (
        "Invalid method. Allowed values are 'row_number', 'rank', and 'dense_rank'."
    )

    # Input validation for n
    assert isinstance(n, int) and n > 0, "n must be a positive, non-zero integer"

    # Add '_dummy_column' if partition_by is not provided
    if partition_by is None:
        if "_dummy_column" in df.columns:
            raise ValueError(
                (
                    "DataFrame already contains '_dummy_column', "
                    "cannot add dummy partition column."
                )
            )
        partition_by = ["_dummy_column"]
        df = df.withColumn("_dummy_column", F.lit(1))

    # Adjust the window specification for ordering in ascending order
    window_spec = Window.partitionBy(*partition_by)
    if order_by is not None:
        window_spec = window_spec.orderBy(*order_by)

    # Add row index column based on the window specification and method
    if method == "row_number":
        df = df.withColumn(index_column_name, F.row_number().over(window_spec))
    elif method == "rank":
        df = df.withColumn(index_column_name, F.rank().over(window_spec))
    elif method == "dense_rank":
        df = df.withColumn(index_column_name, F.dense_rank().over(window_spec))

    # Filter for the first 'n' rows after ordering in ascending order
    df = df.filter(F.col(index_column_name) <= n)

    # Drop unnecessary columns
    if not return_index_column:
        df = df.drop(index_column_name)
    if partition_by == ["_dummy_column"]:
        df = df.drop("_dummy_column")

    return df


def first_row(
    df,
    n=1,
    partition_by=None,
    order_by=None,
    return_index_column=False,
    index_column_name="row_index",
) -> DataFrame:
    """Returns the first N rows per partition by sort order.

    Returns the first `n` rows per partition in a PySpark DataFrame,
    ordered by the specified columns. Wrapper for `select_top_rows`
    with `method='row_number'`.

    Args:
        df (DataFrame): The PySpark DataFrame.
        n (int): Number of rows to retain per partition.
        partition_by (list, optional): Columns to partition by.
        order_by (list, optional): Columns to sort within partitions.
        return_index_column (bool): Include row index column. Defaults to False.
        index_column_name (str): Name for the index column. Defaults to 'row_index'.

    Note:
        When using `order_by`, nulls are treated as smallest by default.
        Use `asc_nulls_last()`, `desc_nulls_first()`, etc., for explicit behavior.

    Returns:
        DataFrame: Filtered DataFrame with top `n` rows per partition.

    Example:
        >>> data = [("A", 1), ("A", 2), ("B", 3)]
        >>> df = spark.createDataFrame(data, ["group", "value"])
        >>> first_row(df, n=1, partition_by=["group"], order_by=["value"]).show()
    """
    df = select_top_rows(
        df,
        method="row_number",
        n=n,
        partition_by=partition_by,
        order_by=order_by,
        return_index_column=return_index_column,
        index_column_name=index_column_name,
    )

    return df


def first_rank(
    df,
    n=1,
    partition_by=None,
    order_by=None,
    return_index_column=False,
    index_column_name="rank_index",
) -> DataFrame:
    """Returns rows with the top N ranks per partition.

    Returns rows corresponding to the first `n` ranks per partition
    in a PySpark DataFrame. Wrapper for `select_top_rows` with `method='rank'`.

    Args:
        df (DataFrame): The PySpark DataFrame.
        n (int): Number of ranks to retain per partition.
        partition_by (list, optional): Columns to partition by.
        order_by (list, optional): Columns to sort within partitions.
        return_index_column (bool): Include rank index column. Defaults to False.
        index_column_name (str): Name for the index column. Defaults to 'rank_index'.

    Note:
        PySpark's `orderBy` treats nulls as smallest by default.
        Use `asc_nulls_last()` or `desc_nulls_first()` to control null ordering.

    Returns:
        DataFrame: Filtered DataFrame with top `n` ranked rows per partition.

    Example:
        >>> data = [("A", 1), ("A", 1), ("B", 2)]
        >>> df = spark.createDataFrame(data, ["group", "value"])
        >>> first_rank(df, n=1, partition_by=["group"], order_by=["value"]).show()
    """
    df = select_top_rows(
        df,
        method="rank",
        n=n,
        partition_by=partition_by,
        order_by=order_by,
        return_index_column=return_index_column,
        index_column_name=index_column_name,
    )

    return df


def first_dense_rank(
    df,
    n=1,
    partition_by=None,
    order_by=None,
    return_index_column=False,
    index_column_name="dense_rank_index",
) -> DataFrame:
    """Returns rows with the top N dense ranks per partition.

    Selects rows with the first 'n' dense ranks per partition based on ordering.
    Wrapper around select_top_rows() with method='dense_rank'.

    Args:
        df (pyspark.sql.DataFrame): Input DataFrame.
        n (int): Number of dense ranks to retain.
        partition_by (list, optional): Columns to partition by.
        order_by (list, optional): Columns to order within each partition.
        return_index_column (bool): Include dense rank column. Defaults to False.
        index_column_name (str): Name of the rank column.
            Defaults to 'dense_rank_index'.

    Note:
        If order_by contains nulls, use asc_nulls_last(), desc_nulls_first(),
        nulls_first(), or nulls_last() to control null sort behavior.

    Returns:
        pyspark.sql.DataFrame: DataFrame with top 'n' dense ranks per partition.

    Example:
        >>> data = [("A", 1), ("A", 1), ("A", 2), ("A", 3),
        ...         ("B", 4), ("B", 5), ("B", 6)]
        >>> df = spark.createDataFrame(data, ["group", "value"])
        >>> result = first_dense_rank(
        ...     df, n=2, partition_by=["group"], order_by=["value"],
        ...     return_index_column=True, index_column_name="dense_rank_index"
        ... )
        >>> result.show()
    """
    df = select_top_rows(
        df,
        method="dense_rank",
        n=n,
        partition_by=partition_by,
        order_by=order_by,
        return_index_column=return_index_column,
        index_column_name=index_column_name,
    )

    return df
