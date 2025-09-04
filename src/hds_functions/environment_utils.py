"""Utilities for environment setup in Databricks notebooks.

These helpers support consistent path resolution and environment setup in shared
workspaces where notebook locations and relative paths can vary.

Functions:
    - get_spark_session: Initialize and return a SparkSession.
    - resolve_path: Construct paths relative to the project root.
    - find_project_folder: Recursively find project root by searching for a marker file.
"""

import os

import pkg_resources
from pyspark.sql import SparkSession


def get_spark_session():
    """Create or get an existing SparkSession.

    Initializes a SparkSession with a default app name if none exists,
    otherwise returns the current active SparkSession.

    Returns:
        SparkSession: The active SparkSession object.

    Example:
        >>> spark = get_spark_session()
    """
    spark_session = SparkSession.builder.appName("SparkSession").getOrCreate()

    return spark_session


def resolve_path(path: str, repo: str = None) -> str:
    """Resolve a file path, handling absolute, relative, and repo-based paths.

    Args:
        path (str): File path; absolute, relative (starts with './'), or repo-relative.
        repo (str, optional): Repository name if path is relative within a repo.

    Returns:
        str: Resolved absolute file path.
    """
    # Check path type combinations are valid; only one of the allowed cases is true
    assert (
        (os.path.isabs(path) and repo is None)
        or (path.startswith("./") and repo is None)
        or (repo is not None and not (path.startswith("./") or os.path.isabs(path)))
    ), (
        "Specify either an absolute path, a relative path with './', "
        "or a path within a repo, not a combination of them."
    )

    if path.startswith("./"):
        # Get project root folder from environment for relative paths
        project_folder = os.environ.get("PROJECT_FOLDER", None)
        # Ensure PROJECT_FOLDER is set, else raise informative error
        assert project_folder is not None, (
            "Environment variable 'PROJECT_FOLDER' not set. "
            "Run './project_setup' at notebook start."
        )
        # Join project root with relative path (remove './' prefix)
        resolved_path = os.path.join(project_folder, path[2:])
    elif repo is not None:
        # Use pkg_resources to get absolute path inside the specified repo
        resolved_path = pkg_resources.resource_filename(repo, path)
    else:
        # Absolute path case, return as is
        resolved_path = path

    return resolved_path


def find_project_folder(marker_file=".dbxproj", workspace_prefix="/Workspace") -> str:
    """Locate project root by searching upward from the notebook path for a marker file.

    Args:
        marker_file (str): Filename that identifies project root (default ".dbxproj").
        workspace_prefix (str): Root prefix of notebook path (default "/Workspace").

    Returns:
        str: Absolute path to the project root folder containing the marker file.

    Raises:
        FileNotFoundError: If the marker file isn't found in any parent directory.

    Example:
        Given a notebook path:
            /Workspace/Users/alice/my_project/notebooks/analysis.ipynb

        And a marker file located at:
            /Workspace/Users/alice/my_project/.dbxproj

        Then calling `find_project_folder()` returns:
            '/Workspace/Users/alice/my_project'
    """
    spark = get_spark_session()  # Get or create a SparkSession instance
    dbutils = get_dbutils(spark)  # Retrieve dbutils for notebook utilities

    # Get the notebook context to find current notebook's folder path
    context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()

    # Compose the full notebook folder path including workspace prefix
    notebook_folder = (
        f"{workspace_prefix}{os.path.dirname(context.notebookPath().get())}"
    )

    current_path = notebook_folder  # Start search from the notebook's folder

    while True:
        # If reached root without finding marker, raise error
        if current_path in ("", "/"):
            raise FileNotFoundError(
                f"Marker file '{marker_file}' not found in any parent directories "
                f"of {notebook_folder}."
            )

        try:
            # Check if marker file exists in current directory
            if marker_file in os.listdir(current_path):
                return current_path  # Found project root, return path
        except (FileNotFoundError, PermissionError):
            pass  # Skip inaccessible directories without stopping execution

        # Move up one directory level to continue the search
        current_path = os.path.dirname(current_path)


def get_dbutils(spark: SparkSession):
    """Get a DBUtils instance for Databricks notebook utilities.

    Tries to create DBUtils from the Spark session first; if unavailable,
    falls back to retrieving `dbutils` from IPython user namespace.

    Args:
        spark (SparkSession): Active Spark session.

    Returns:
        dbutils (DBUtils): Databricks utilities instance.

    Raises:
        RuntimeError: If dbutils cannot be found in the environment.

    Example:
        >>> spark = SparkSession.builder.getOrCreate()
        >>> dbutils = get_dbutils(spark)
        >>> ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        >>> print(ctx.notebookPath().get())
    """
    try:
        # Try importing DBUtils from pyspark (Databricks environment)
        from pyspark.dbutils import DBUtils

        # Return a DBUtils instance using the active Spark session
        return DBUtils(spark)
    except ImportError:
        try:
            # Fallback: import IPython to access notebook user namespace
            import IPython

            # Return dbutils from IPython's user namespace if available
            return IPython.get_ipython().user_ns["dbutils"]
        except (KeyError, AttributeError) as err:
            # Raise error if dbutils is not found in either approach
            raise RuntimeError("dbutils is not available in this environment.") from err
