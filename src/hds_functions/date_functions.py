"""Module to parse and convert date instructions into PySpark SQL expressions.

Provides utilities to parse date strings and relative date operations,
validate date literals, and convert date units into day-based expressions
compatible with PySpark.

Functions:
    - parse_date_instruction: Parse date or relative operations to PySpark SQL.
    - convert_date_units_to_days: Convert relative units to day counts in expressions.
    - validate_date_string: Check if string is a valid 'YYYY-MM-DD' date.
"""

import re
from datetime import datetime


def parse_date_instruction(date_string: str) -> str:
    """Parse a date transformation string into a PySpark SQL expression.

    Accepts strings with date literals or relative operations (e.g., add/subtract
    days, weeks, months, years). Converts them into expressions compatible with
    PySpark's `f.expr()` for use in date transformations.

    Args:
        date_string (str): Instruction string defining a date operation.

    Returns:
        str: A PySpark-compatible SQL expression.

    Raises:
        ValueError: If an invalid date literal is supplied.

    Examples:
        >>> parse_date_instruction('2020-01-01')
        "date('2020-01-01')"

        >>> parse_date_instruction('2020-02-30')
        Traceback (most recent call last):
            ...
        ValueError: Invalid date: '2020-02-30'

        >>> parse_date_instruction('index_date + 5 days')
        "index_date + cast(round(5*1) as int)"

        >>> parse_date_instruction('x - 6 weeks')
        "x - cast(round(6*7) as int)"

        >>> parse_date_instruction('index_date + 3 months')
        "index_date + cast(round(3*30) as int)"

        >>> parse_date_instruction('index_date - 2 years')
        "index_date - cast(round(2*365.25) as int)"

        >>> parse_date_instruction('index_date')
        'index_date'

        >>> parse_date_instruction('current_date() + 5 days')
        'current_date() + cast(round(5*1) as int)'
    """
    # Check if date_string is None, if so return 'NULL'
    if date_string is None:
        return "cast(NULL as date)"

    # Check if the instruction is a simple date string
    elif re.match(r"\d{4}-\d{2}-\d{2}", date_string):
        if validate_date_string(date_string):
            return f"date('{date_string}')"
        else:
            raise ValueError(f"Invalid date: {date_string}")

    # Check if the instruction is a transformation expression
    elif any(
        unit in date_string
        for unit in ["day", "days", "week", "weeks", "month", "months", "year", "years"]
    ):
        parsed_expression = convert_date_units_to_days(date_string)
        return parsed_expression

    # Otherwise return original date expression
    else:
        return date_string


def convert_date_units_to_days(date_expression: str) -> str:
    """Convert date units in an expression to days, rounding and casting to int.

    Extracts the numeric value and unit (day, week, month, year) from the input
    expression, converts the unit to days by multiplying with the correct factor
    (1 for day, 7 for week, 30 for month, 365.25 for year), then wraps the result
    in round() and casts it to int.

    Args:
        date_expression (str): Date or transformation expression, e.g.,
            'index_date + 6 months' or 'x - 7.5 weeks'.

    Returns:
        str: Expression with units converted to days and result cast to int.

    Example:
        >>> expr = 'index_date - 2 years, x - 7.5 weeks'
        >>> convert_date_units_to_days(expr)
        'index_date - cast(round(2*365.25) as int), x - cast(round(7.5*7) as int)'
    """
    # Regex to find all "<number> <unit>" patterns
    pattern = r"\b(\d+(?:\.\d+)?)\s*(\w+)\b"
    matches = re.findall(pattern, date_expression)

    unit_to_days = {
        "day": 1,
        "days": 1,
        "week": 7,
        "weeks": 7,
        "month": 30,
        "months": 30,
        "year": 365.25,
        "years": 365.25,
    }

    for number, unit in matches:
        if unit not in unit_to_days:
            raise ValueError(
                f"Invalid unit: {unit}. Use 'day', 'week', 'month', or 'year'."
            )

        converted_expression = f"cast(round({number}*{unit_to_days[unit]}) as int)"

        # Replace original number + unit with converted expression
        date_expression = re.sub(
            rf"\b{number}\s*{unit}\b", converted_expression, date_expression
        )

    return date_expression


def validate_date_string(date_string: str) -> bool:
    """Validate if date_string is a real date in 'YYYY-MM-DD' format.

    Checks that the input string matches the standard date format and
    represents a valid calendar date.

    Args:
        date_string (str): Date string to validate.

    Returns:
        bool: True if valid date in 'YYYY-MM-DD' format, else False.

    Examples:
        >>> validate_date_string('2020-01-01')
        True
        >>> validate_date_string('2020-02-30')
        False
    """
    # Try to parse the string into a date with the given format
    try:
        datetime.strptime(date_string, "%Y-%m-%d")
        return True
    # Parsing failed: invalid date or format
    except ValueError:
        return False
