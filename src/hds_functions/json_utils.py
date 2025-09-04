"""This module provides utilities for reading and writing JSON files in Python."""

import json
import os
from typing import Any, Dict

from .environment_utils import resolve_path


def read_json_file(path: str, repo: str = None) -> Dict[str, Any]:
    """Read a JSON file and return its contents as a dictionary.

    Loads a JSON file from the specified path (optionally within a repo) in binary mode,
    checks for duplicate keys, and returns its contents as a Python dictionary.

    Args:
        path (str): Path to the JSON file: absolute, relative ('./'), or within a repo.
        repo (str, optional): Repo name if using a repo-relative path.

    Returns:
        dict: Parsed JSON content.

    Raises:
        ValueError: If the file contains duplicate keys.

    Examples:
        >>> read_json_file('./relative/path/in/project.json')
        >>> read_json_file('/Workspace/absolute/path.json')
        >>> read_json_file(path='path/in/repo.json', repo='common_repo')
    """

    def check_json_for_duplicate_keys(ordered_pairs):
        """Hook to detect duplicate keys while parsing a JSON object.

        Args:
            ordered_pairs: A list of key-value pairs parsed from a JSON object.

        Returns:
            dict: The reconstructed JSON object as a dictionary.

        Raises:
            ValueError: If duplicate keys are found in the object.
        """
        d = {}
        for k, v in ordered_pairs:
            if k in d:
                raise ValueError(
                    f"JSON file '{resolved_path}' contains duplicate key: {k}"
                )
            else:
                d[k] = v
        return d

    # Resolve the path
    resolved_path = resolve_path(path, repo)

    # Load JSON file and check for duplicate keys
    with open(resolved_path) as json_file:
        json_dict = json.load(
            json_file, object_pairs_hook=check_json_for_duplicate_keys
        )

    return json_dict


def write_json_file(
    data: Dict[str, Any], path: str, repo: str = None, indent: int = 4
) -> None:
    """Write a dictionary to a JSON file at the given path.

    Saves the dictionary as a formatted JSON file, optionally within a repo, using
    the specified indentation for readability.

    Args:
        data (dict): Dictionary to write.
        path (str): File path (absolute, relative, or within a repo).
        repo (str, optional): Repo name if using a repo-relative path.
        indent (int, optional): Number of spaces for indentation. Defaults to 4.

    Returns:
        None

    Examples:
        >>> data = {"key1": "value1", "key2": "value2"}
        >>> write_json_file(data, "./in_project_folder.json")
        >>> write_json_file(data, "/Workspace/absolute_path.json")
        >>> write_json_file(data, path="in/shared/repo.json", repo="common_repo")
    """
    # Resolve the path
    resolved_path = resolve_path(path, repo)

    # Check if the directory exists
    directory = os.path.dirname(resolved_path)
    if not os.path.exists(directory):
        raise ValueError(f"Directory '{directory}' does not exist.")

    # Write file
    with open(resolved_path, "w") as fp:
        json.dump(data, fp, indent=indent)
