"""Module for constructing and managing cohort tables.

Description:
    This module provides functions for constructing and managing cohort tables.
    It includes functionality for filtering cohorts based on specified inclusion
    criteria and generating flowchart tables to visualize the application of these
    criteria.

Functions:
    - apply_inclusion_criteria: Filters cohort DataFrame based on inclusion criteria.
        Optionally creates flowchart.
    - create_inclusion_columns: Creates flag columns based on inclusion criteria
    - create_inclusion_flowchart: Logs the changes in rows and individuals for each
        step in inclusion criteria
"""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.utils import AnalysisException

from .environment_utils import get_spark_session
from .table_management import save_table


def apply_inclusion_criteria(
    cohort: DataFrame,
    inclusion_criteria: dict[str, str],
    flowchart_table: str = None,
    row_id_col: str = "row_id",
    person_id_col: str = "person_id",
    drop_inclusion_flags: bool = True,
) -> DataFrame:
    """Apply inclusion criteria to the cohort and optionally generate a flowchart table.

    Args:
        cohort (DataFrame): Input cohort data.
        inclusion_criteria (dict[str, str]): Mapping of flag columns to SQL expressions.
        flowchart_table (str, optional): Flowchart table key. Defaults to None.
        row_id_col (str, optional): Row ID column name. Defaults to "row_id".
        person_id_col (str, optional): Person ID column name. Defaults to "person_id".
        drop_inclusion_flags (bool, optional): Drop flag columns after filtering.
            Defaults to True.

    Returns:
        DataFrame: Filtered cohort DataFrame.

    Example:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.getOrCreate()
        >>> data = [(1, "id_001", 30), (2, "id_002", 70), (3, None, 40)]
        >>> cohort = spark.createDataFrame(data, ["row_id", "person_id", "age"])
        >>> criteria = {"valid_id": "person_id IS NOT NULL", "age_ok": "age < 65"}
        >>> filtered = apply_inclusion_criteria(cohort, criteria)
        >>> filtered.show()
        +------+---------+---+
        |row_id|person_id|age|
        +------+---------+---+
        |     1|   id_001| 30|
        +------+---------+---+
    """
    # Validate that inclusion criteria keys exist and are valid for the cohort DataFrame
    validate_inclusion_criteria(cohort, inclusion_criteria)

    # Validate cohort columns to ensure no forbidden columns are present or conflicting
    validate_cohort_columns(cohort, inclusion_criteria, row_id_col, person_id_col)

    # Add columns to cohort DataFrame flagging rows that meet each inclusion criterion
    cohort_flagged = create_inclusion_columns(cohort, inclusion_criteria)

    # If flowchart_table key is provided, generate and save the flowchart
    if flowchart_table:
        flowchart = create_inclusion_flowchart(
            cohort_flagged, inclusion_criteria, row_id_col, person_id_col
        )
        save_table(df=flowchart, table=flowchart_table)

    # Filter the cohort to only rows that meet all inclusion criteria (include == True)
    cohort_filtered = cohort_flagged.filter(F.col("include"))

    # Optionally drop the inclusion flag columns and intermediate criteria columns
    if drop_inclusion_flags:
        columns_to_drop = (
            [f"criteria_{i}" for i in range(len(inclusion_criteria) + 1)]
            + list(inclusion_criteria.keys())
            + ["include"]
        )
        cohort_filtered = cohort_filtered.drop(*columns_to_drop)

    return cohort_filtered


def create_inclusion_columns(
    cohort: DataFrame, inclusion_criteria: dict[str, str]
) -> DataFrame:
    """Add inclusion criteria flag columns to the cohort DataFrame.

    Args:
        cohort (DataFrame): Input cohort DataFrame.
        inclusion_criteria (dict[str, str]): Mapping of flag column names to SQL
            expressions.

    Returns:
        DataFrame: Cohort DataFrame augmented with
            - `criteria_*` boolean columns for each inclusion criterion, and
            - an `include` column indicating if all criteria are met.
    """
    # Add a column for each inclusion criterion based on its SQL expression
    for column_name, sql_expression in inclusion_criteria.items():
        cohort = cohort.withColumn(column_name, F.expr(sql_expression))

    # Replace nulls in inclusion columns with False (missing does not meet criteria)
    cohort = cohort.fillna(False, list(inclusion_criteria.keys()))

    # Start criteria chain with a column always True (base case for cumulative AND)
    cohort = cohort.withColumn("criteria_0", F.lit(True))

    # Create cumulative AND columns to check if all criteria up to current are True
    for index, column_name in enumerate(inclusion_criteria.keys(), start=1):
        cohort = cohort.withColumn(
            f"criteria_{index}", F.col(f"criteria_{index - 1}") & F.col(column_name)
        )

    # Final 'include' column is True only if all criteria are met
    cohort_flagged = cohort.withColumn(
        "include", F.col(f"criteria_{len(inclusion_criteria)}")
    )

    return cohort_flagged


def create_inclusion_flowchart(
    cohort_flagged: DataFrame,
    inclusion_criteria: dict[str, str],
    row_id_col: str = "row_id",
    person_id_col: str = "person_id",
) -> DataFrame:
    """Generate a flowchart DataFrame summarising inclusion criteria effects.

    Args:
        cohort_flagged (DataFrame): Cohort with 'criteria_*' and 'include' flag columns.
        inclusion_criteria (dict[str, str]): Mapping of criteria column names to SQL
            expressions.
        row_id_col (str): Column name for row IDs. Defaults to "row_id".
        person_id_col (str): Column name for person IDs. Defaults to "person_id".

    Returns:
        DataFrame: Flowchart showing counts of rows and distinct persons passing each
            criterion.
    """
    spark = get_spark_session()  # Get active Spark session
    criteria_columns = [f"criteria_{i}" for i in range(len(inclusion_criteria) + 1)]

    # Create a DataFrame describing criteria with their names, descriptions, and exprs
    df_inclusion_criteria = spark.createDataFrame(
        [("criteria_0", "Original table", "")]
        + [
            (f"criteria_{i + 1}", k, v)
            for i, (k, v) in enumerate(inclusion_criteria.items())
        ],
        ["criteria", "description", "expression"],
    )

    id_cols = [row_id_col, person_id_col]
    flowchart_selected = cohort_flagged.select(id_cols + criteria_columns)

    # Unpivot criteria columns to rows, with 'passed' bool per criterion per row
    flowchart_unpivoted = flowchart_selected.unpivot(
        ids=id_cols,
        values=criteria_columns,
        variableColumnName="criteria",
        valueColumnName="passed",
    )

    # Count rows and distinct persons passing each criterion
    flowchart_aggregated = flowchart_unpivoted.groupBy("criteria").agg(
        F.count(F.when(F.col("passed"), 1)).alias("n_row"),
        F.countDistinct(F.when(F.col("passed"), F.col(person_id_col))).alias(
            "n_distinct_id"
        ),
    )

    # Join with descriptions & expressions for criteria
    flowchart_with_desc = flowchart_aggregated.join(
        F.broadcast(df_inclusion_criteria), on="criteria", how="left"
    )

    # Extract numeric index from criteria for ordering
    flowchart_with_index = flowchart_with_desc.withColumn(
        "criteria_index", F.regexp_extract("criteria", r"\d+", 0).cast("int")
    )

    window_spec = Window.orderBy("criteria_index")  # Window for lag calculations

    # Calculate excluded rows and ids between criteria steps
    flowchart_difference = flowchart_with_index.withColumn(
        "excluded_rows",
        (F.lag("n_row", 1).over(window_spec) - F.col("n_row")).cast("int"),
    ).withColumn(
        "excluded_ids",
        (F.lag("n_distinct_id", 1).over(window_spec) - F.col("n_distinct_id")).cast(
            "int"
        ),
    )

    # Select final columns and order by criteria index
    flowchart_final = flowchart_difference.select(
        "criteria_index",
        "criteria",
        "description",
        "expression",
        "n_row",
        "n_distinct_id",
        "excluded_rows",
        "excluded_ids",
    ).orderBy("criteria_index")

    return flowchart_final


def validate_inclusion_criteria(
    cohort: DataFrame, inclusion_criteria: dict[str, str]
) -> None:
    """Validate the structure of inclusion criteria.

    Args:
        cohort (DataFrame): Input cohort DataFrame.
        inclusion_criteria (dict[str, str]): Mapping of column names to SQL expressions.

    Raises:
        TypeError: If inclusion_criteria is not a dict or if any SQL expression is not a
            string.
    """
    # Check if inclusion_criteria is a dictionary
    if not isinstance(inclusion_criteria, dict):
        raise TypeError(
            "The inclusion_criteria must be a dictionary where keys are criteria "
            "column names and values are SQL expressions."
        )

    # Check each SQL expression is a string
    for key, value in inclusion_criteria.items():
        if not isinstance(value, str):
            raise TypeError(
                f"The SQL expression for inclusion criteria '{key}' must be a string, "
                f"but got {type(value).__name__}."
            )


def validate_cohort_columns(
    cohort: DataFrame,
    inclusion_criteria: dict[str, str],
    row_id_col: str,
    person_id_col: str,
) -> None:
    """Validate that required columns exist and no conflicting columns are present.

    Checks that the cohort DataFrame includes the row and person ID columns and
    that it does not contain columns that conflict with 'criteria_*', 'include',
    or any keys from inclusion_criteria.

    Args:
        cohort (DataFrame): Input DataFrame containing cohort data.
        inclusion_criteria (dict[str, str]): Inclusion criteria keys to check.
        row_id_col (str): Column name for row identifiers.
        person_id_col (str): Column name for person identifiers.

    Raises:
        ValueError: If conflicting column names are found in the cohort.
        AnalysisException: If row_id_col or person_id_col are missing.
    """
    # Get all column names as a set for fast lookup
    cohort_columns = set(cohort.columns)

    # Start with any existing 'criteria_*' and 'include' columns
    forbidden_columns = {
        col for col in cohort_columns if col.startswith("criteria_")
    } | {"include"}

    # Add keys from inclusion_criteria to the forbidden set
    forbidden_columns |= set(inclusion_criteria.keys())

    # Check for overlap between cohort columns and forbidden set
    conflicting_columns = forbidden_columns.intersection(cohort_columns)
    if conflicting_columns:
        conflict_str = ", ".join(conflicting_columns)
        raise ValueError(
            f"The cohort DataFrame contains conflicting columns: {conflict_str}"
        )

    # Check that both row_id_col and person_id_col are present
    missing_columns = [
        col for col in (row_id_col, person_id_col) if col not in cohort_columns
    ]
    if missing_columns:
        raise AnalysisException(
            f"Missing required columns: {', '.join(missing_columns)}"
        )
