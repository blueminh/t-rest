import numpy as np
import pandas as pd
import typing as tp

def _convert_encodetext_to_tabular_data(text: tp.List[str], df_gen: pd.DataFrame) -> pd.DataFrame:
    """ Converts the sentences back to tabular data

    Args:
        text: List of the tabular data in text form
        df_gen: Pandas DataFrame where the tabular data is appended

    Returns:
        Pandas DataFrame with the tabular data from the text appended
    """
    columns = df_gen.columns.to_list()

    # Convert text to tabular data
    for t in text:
        features = t.split(",")
        td = dict.fromkeys(columns)

        for f in features:
            values = f.strip().split(" : ")
            if values[0] in columns and not td[values[0]]:
                try:
                    td[values[0]] = [values[1]]
                except IndexError:
                    pass

        # # Transform all features back to tabular data
        # for f in features:
        #     values = f.strip().split(" is ")
        #     if values[0] in columns and not td[values[0]]:
        #         td[values[0]] = [values[1]]

        df_gen = pd.concat([df_gen, pd.DataFrame(td)], ignore_index=True, axis=0)
    return df_gen


def _encode_row_partial(row, shuffle=True):
    """ Function that takes a row and converts all columns into the text representation that are not NaN."""
    num_cols = len(row.index)
    if not shuffle:
        idx_list = np.arange(num_cols)
    else:
        idx_list = np.random.permutation(num_cols)

    lists = ", ".join(
        sum([[f"{row.index[i]} is {row[row.index[i]]}"] if not pd.isna(row[row.index[i]]) else [] for i in idx_list],
            []))
    return lists
    # Now append first NaN attribute


def _get_random_missing(row):
    """ Return a random missing column or None if all columns are filled. """
    nans = list(row[pd.isna(row)].index)
    return np.random.choice(nans) if len(nans) > 0 else None


def _partial_df_to_promts(partial_df: pd.DataFrame):
    """ Convert DataFrame with missingvalues to a list of starting promts for GReaT
        Args:
        partial_df: Pandas DataFrame to be imputed where missing values are encoded by NaN.

    Returns:
        List of strings with the starting prompt for each sample.
    """
    encoder = lambda x: _encode_row_partial(x, True)
    res_encode = list(partial_df.apply(encoder, axis=1))
    res_first = list(partial_df.apply(_get_random_missing, axis=1))

    # Edge case: all values are missing, will return empty string which is not supported.
    # Use first attribute as starting prompt.
    # default_promt = partial_df.columns[0] + " is "
    res = [((enc + ", ") if len(enc) > 0 else "") + (fst + " is" if fst is not None else "") for enc, fst in
           zip(res_encode, res_first)]
    return res
