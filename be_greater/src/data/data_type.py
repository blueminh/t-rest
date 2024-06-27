import typing
from collections import defaultdict

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype, is_numeric_dtype

Column = typing.TypeVar("Column", bound=str)
Precision = typing.TypeVar("Precision", bound=int)

PRECISION_LOOKUP: typing.Dict[str, typing.Dict[str, typing.Optional[int]]] = {
    "intrusion": defaultdict(lambda: 2),
    "king": defaultdict(lambda: 3,
                        bathrooms=2,
                        floors=1,
                        lat=4,
                        long=3
                        ),
    "loan": defaultdict(lambda: 1),
    "adult": defaultdict(lambda: None)
}

def format_int(x: int, negative: bool, decimals: int = 0):
    """Formatting function to convert an integer `x` into a string representation that matches the requested formatting.

    Args:
        negative (bool): If `True`, a sign is pre-pended for the input
        decimals (int): Prepends the formatted number with decimals - log_10(x) `0`'s

    Returns
        str: Formatted string representation of the inputed `x`.
    """
    if negative:
        return f"{'+' if x > -1 else '-'}{abs(x):0{decimals}}"
    else:
        return f"{x:0{decimals}}"


def convert_number_int(values, negative = False, extrema = None):
    """
    Method for converting a column of integers to their string representation to preprocess data.
    """
    # assert is_integer_dtype(values)
    extrema = extrema or values.abs().max()
    # Get base 10 encoding
    # +1 for +/- sign
    number_of_numbers = int(np.log10(extrema) + 1)
    negative = negative or values.min() < 0
    return values.apply(format_int, args=[negative, number_of_numbers])


def format_float(x, precision, negative, decimals):
    """Formatting function to convert a float `x` into a string representation that matches the requested formatting.

    Args:
        negative (bool): If `True`, a sign is pre-pended for the input
        decimals (int): Prepends the formatted number with decimals - log_10(x) `0`'s

    Returns
        str: Formatted string representation of the inputed `x`.
    """
    if negative:
        return  f"{'+' if x >= 0.0 else '-'}{abs(x):0{decimals+precision+1}.{precision}f}"
    else:
        return  f"{x:0{decimals+precision+1}.{precision}f}"


def convert_number_float(values, decimals = 3, negative = False, extrema = None):
    """Helper method to convert a real number to a fixed length string representation. E.g. -23.23252 to -0023.233.
    """
    # assert is_float_dtype(values)
    extrema = extrema or values.abs().max()
    # Get base 10 encoding
    # +1 for +/- sign
    # get the number of decimals before the period. Note that we add 1 after flooring to account for the edge case of the
    # first in an order of magnitude suchas 100, 1000, etc.
    encoding_length = int(np.ceil(np.log10(np.floor(extrema) + 1))) # + decimals
    negative = negative or values.min() < 0

    return values.apply(format_float, args=[decimals, negative, encoding_length])


def stringify_dataframe(df: pd.DataFrame, precision: int=3, precision_map:typing.Optional[typing.Dict[str, int]]=None):
    """Helper method to generate a stringified representation of a dataframe. Currently, supports mapping integer and
    floating points numbers to be converted to fixed length 'strings'.

    Args:
        df (pd.DataFrame): DataFrame to convert to fixed string content.
        precision (int, *, 3): Integer indicator for default precision of columns if no precision_map is provided.
        precision_map (dict, *): Optional precision map providing (integer based) precision for fractional component of
            continuous (floating point) numbers.

    Returns:
        DataFrame with columns mapped to *fixed* lenght (string representation) numbers.
    """
    strifified_df = pd.DataFrame(columns=df.columns)
    for column, dtype in zip(df, df.dtypes):
        if is_integer_dtype(dtype):
            strifified_df[column] = convert_number_int(df[column])
        elif is_float_dtype(dtype):
            float_precision = precision_map.get(column, precision)
            strifified_df[column] = convert_number_float(df[column], float_precision)
        else:
            strifified_df[column] = df[column].copy()

    return strifified_df


def get_precision(dataset: str):
    """Helper function to get precision of a benchmark dataset by name.

    Args:
        dataset (str): Name of the dataset to get precision for.
    """
    return PRECISION_LOOKUP[dataset]

def convert_dataframe(
        df: pd.DataFrame,
        dataset: str = None,
        conversion_map: typing.Dict[Column, typing.Optional[Precision]] = None
) -> typing.Tuple[pd.DataFrame, typing.Dict[str, typing.Callable]]:
    """Convert dataframe from pre-formatted to formatted representation  of each columns' values.

    Args:
        df (pd.DataFrame): Dataframe to re-format.
        dataset (st): Dataset name.
        conversion_map (dict, *, None): Optional conversion map for testing purposes or overwriting default precision
            map.

    Returns:
        Re-formatted dataset.
        Precision map of corresponding map.
    """
    conversion_map = conversion_map or PRECISION_LOOKUP[dataset]
    df_string = stringify_dataframe(df=df, precision_map=conversion_map)

    for column, dtype in zip(df, df.dtypes):
        if is_numeric_dtype(dtype):
            df_string[column] = df_string[column].apply(list)
        else:
            df_string[column] = df_string[column].apply(lambda x: [str(x)])

    return df_string, conversion_map
