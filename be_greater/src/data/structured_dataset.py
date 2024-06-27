from dataclasses import dataclass
import random

import transformers
import typing as tp
import pandas as pd
from pandas import DataFrame
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

from be_great.great_dataset import GReaTDataset
from src.data.data_type import convert_dataframe

PrecisionMap = tp.Dict[str, tp.Callable]


class StructuredDataset(GReaTDataset):
    """ GReaT Dataset

    The GReaTDataset overwrites the _getitem function of the HuggingFace Dataset Class to include the permutation step.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer from HuggingFace
    """

    def _getitem(self, key: tp.Union[int, slice, str], decoded: bool = True, **kwargs) -> tp.Union[tp.Dict, tp.List]:
        """ Get Item from Tabular Data

        Get one instance of the tabular data, permuted, converted to text and tokenized.
        """
        # If int, what else?
        row = self._data.fast_slice(key, 1)

        shuffle_idx = list(range(row.num_columns))
        random.shuffle(shuffle_idx)

        structured_representation = [f"{row.column_names[0]} is "] + row.columns[0].to_pylist()[0]
        for indx in shuffle_idx[1:]:
            structured_representation += [f", {row.column_names[indx]} is "] + row.columns[indx].to_pylist()[0]

        return structured_representation

    def __getitems__(self, keys: tp.Union[int, slice, str, list]):
        if isinstance(keys, list):
            return self.tokenizer.batch_encode_plus(
                    [self._getitem(key) for key in keys],
                    is_split_into_words=True,
                    padding=True,
                    return_tensors='pt'
            )
        else:
            return self.tokenizer.encode(self._getitem(keys))


@dataclass
class GReaTDataCollator(DataCollatorWithPadding):
    """ GReaT Data Collator

    Overwrites the DataCollatorWithPadding to also pad the labels and not only the input_ids
    """

    def __call__(self, features: tp.List[tp.Dict[str, tp.Any]]):
        batch = self.tokenizer.pad(
                features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch


def structured_dataset_from_df(
        dataset: str,
        df: pd.DataFrame,
        tokenizer: transformers.PreTrainedTokenizer,
        **conversion_kwargs
) -> tp.Tuple[DataFrame, pd.DataFrame, StructuredDataset, PreTrainedTokenizer, PrecisionMap]:
    """Helper function to convert a Dataframe to a Structured Dataset with appropriate mapping applied
    Args:
        dataset (str): Dataset name.
        df (pd.DataFrame): Dataset loaded into DataFrame.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used for dataset.

    Returns:
        Original DataFrame
        Re-structured DataFrame
        StructuredDataset object with dataset mapped to fixed-length encoding.
        Tokenizer for training and generation.
        Precision map for each column for generation and masking.
    """

    df_modified, precision_map = convert_dataframe(df, dataset, conversion_kwargs)
    dataset = StructuredDataset.from_pandas(df_modified, preserve_index=False)
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.pad_token_id = 50256
    dataset.set_tokenizer(tokenizer)
    return df, df_modified, dataset, tokenizer, precision_map
