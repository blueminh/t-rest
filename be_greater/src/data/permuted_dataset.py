import abc
import logging
import random
import typing as tp
from typing import Union, Dict, List, Optional

import jinja2
import pandas as pd
import torch

from be_great import GReaT
from be_great.great_dataset import GReaTDataset
from be_great.great_start import GReaTStart, _pad_tokens
from datasets import DatasetInfo, NamedSplit
from pyarrow import Table


class GReatPrompter(abc.ABC):
    def __init__(
            self,
            start_prompt='{% for v in values %}{{ columns[loop.index0] }} is {{ v }}{% if not loop.last %}, {% endif %}{% endfor %}',
            REQUIRES_ALL_COLUMNS: bool = False):
        self.start_prompt = start_prompt
        self.template = jinja2.Template(start_prompt)
        self.REQUIRES_ALL_COLUMNS = REQUIRES_ALL_COLUMNS

    def prompt(self, row: Table, ordering: Optional[List[int]] = None):
        ''' Transform Row to re-prompted version.
        :param ordering: Optional ordering of columns.
        :param row: Row to be transfromed
        :return: Textual representation of row
        '''
        prompt = self.template.render(row=row, columns=[row.column_names[i] for i in ordering],
                                      values=[row.columns[i].to_pylist()[0] for i in ordering])
        return prompt


class ContextPrompter(GReatPrompter):
    ''' Extended Prompter for GReaT'''

    def __init__(self, start_prompt='Row with {{ columns|length }} values: '
                                    '{% for v in values %}{{ columns[loop.index0] }} is {{ v }}{% if not loop.last %}, {% endif %}{% endfor %}',
                 column_aware = False):
        super(ContextPrompter, self).__init__(start_prompt)
        self.column_aware = column_aware


    def prompt(self, row: Table, ordering: Optional[List[int]] = None):
        ''' Transform Row to re-prompted version.
        :param ordering: Optional ordering of columns.
        :param row: Row to be transformed
        :return: Textual representation of row
        '''
        if self.column_aware:
            # This will yield a reduced
            indices = {row.column_names[i]: value for indx, i in enumerate(ordering) if (value:= row.columns[i].to_pylist()[0]) is not None}
            prompt = self.template.render(row=row, columns=indices.keys(), values=indices.values())

        else:
            # columns = [row.column_names[i] for i in ordering]
            # values =
            indices = {row.column_names[i]: row.columns[i].to_pylist()[0] for indx, i in enumerate(ordering)}


            # TODO: Write test for correct rendering of prompts depending on missingness
            prompt = self.template.render(row=row, columns=list(indices.keys()),
                                          values=[row.columns[i].to_pylist()[0] for i in ordering])
        return prompt




class ContextReorderedPrompter(GReatPrompter):
    ''' Extended Prompter for GReaT'''

    def __init__(self, start_prompt='Row with {{ columns|length }} values: '
                                    '{% for c in columns %}{{ c }}{% if not loop.last %}, {% endif %}{% endfor %}: '
                                    '{% for v in values %}{{ v }}{% if not loop.last %}, {% endif %}{% endfor %}',
                 aware = True):
        """

        :param start_prompt:
        :param aware: Whehter or not the prompter should be aware of the number of available thingies.
        """
        super(ContextReorderedPrompter, self).__init__(start_prompt, REQUIRES_ALL_COLUMNS=True)
        self.aware = aware

    def prompt(self, row: Table, ordering: Optional[List[int]] = None):
        ''' Transform Row to re-prompted version. Overwrites default behavior as the number fo
        :param ordering: Optional ordering of columns.
        :param row: Row to be transfromed
        :return: Textual representation of row
        '''
        if self.aware:
            indices = {row.column_names[i]: value for indx, i in enumerate(ordering) if (value:= row.columns[i].to_pylist()[0]) is not None}
            # columns = [row.column_names[i] for i in ordering]
            # values =
            return self.template.render(row=row, columns=indices.keys(), values=indices.values())
        else:
            return super(ContextReorderedPrompter, self).prompt(row, ordering)

class PromptedGReaTDataset(GReaTDataset):
    """ Adapted from GReaT Dataset

    The GReaTDataset overwrites the _getitem function of the HuggingFace Dataset Class to include the permutation step.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer from HuggingFace
    """

    def __init__(self, arrow_table: Table, info: Optional[DatasetInfo] = None, split: Optional[NamedSplit] = None,
                 indices_table: Optional[Table] = None, fingerprint: Optional[str] = None):

        super(PromptedGReaTDataset, self).__init__(arrow_table, info, split, indices_table, fingerprint)
        self.prompter = GReatPrompter()

    def set_prompter(self, prompter: GReatPrompter):
        """ Method to overwrite default prompting behavior of GReaT datasets"""
        assert prompter
        assert isinstance(prompter, GReatPrompter)
        if isinstance(self.prompter, GReatPrompter):
            logging.info(f"Overwriting default GReaT prompter with: {type(prompter)}")
        else:
            logging.warning(f"Overwriting non-default {type(self.prompter)} prompter with: {type(prompter)}. Are you "
                            f"sure that you want to set the prompter twice?")

        self.prompter = prompter

    def _getitem(self, key: Union[int, slice, str], decoded: bool = True, **kwargs) -> Union[Dict, List]:
        """ Get Item from Tabular Data

        Get one instance of the tabular data, permuted, converted to text and tokenized.
        """
        # If int, what else?
        row = self._data.fast_slice(key, 1)

        shuffle_idx = list(range(row.num_columns))
        random.shuffle(shuffle_idx)
        # Rather than $column_i is P{
        prompt = self.prompter.prompt(row, shuffle_idx)

        tokenized_text = self.tokenizer(prompt)
        return tokenized_text


class CompoundedGreatSampler(GReaTStart):
    """ Compoa
    """

    def __init__(self, tokenizer, prompter: GReatPrompter, great: GReaT):
        super(CompoundedGreatSampler, self).__init__(tokenizer)
        self.great_start = great._get_start_sampler('', None)
        self.prompter = prompter
        self.columns = great.columns
        self.num_columns = great.num_cols
        self.fake_row = None
        self.prepare_start_prompt()

    def get_start_words(self, n_samples: int) -> tp.List[tp.List[int]]:
        return self.great_start.get_start_words(n_samples)

    def prepare_start_prompt(self):
        temp_dataset = GReaTDataset.from_pandas(pd.DataFrame(columns=self.columns))
        self.fake_row = temp_dataset._data.fast_slice()

    def get_start_tokens(self, n_samples: int) -> tp.List[tp.List[int]]:
        """ Overwrites default behavior to get start_tokens, by leveraging the Dataset specific one, and using the
        provided prompter to generate the corresponding text.
        :param n_samples: Number of samples to generate
        :return:
        """
        start_words = self.great_start.get_start_words(n_samples)
        # Use last item / column by default
        if self.prompter.REQUIRES_ALL_COLUMNS:
            # We need all categories in this case.
            start_text = [
                self.prompter.template.render(
                    row=self.fake_row,
                    columns=self.columns[::-1],
                    values=[s] if not isinstance(s, list) else s[::-1])
                for s in start_words]
        else:
            start_text = [
                self.prompter.template.render(
                    row=self.fake_row,
                    columns=self.columns[-1:],
                    values=[s])
                for s in start_words]
        start_tokens = torch.tensor( _pad_tokens(self.tokenizer(start_text)['input_ids']), dtype=torch.long)
        return start_tokens


class StructuredGreatSampler(GReaTStart):
    """ Compoa
    """

    def __init__(self, tokenizer, great: GReaT, value_mapper):
        super(StructuredGreatSampler, self).__init__(tokenizer)
        self.great_start = great._get_start_sampler('', None)
        self.value_mapper = value_mapper
        self.columns = great.columns
        self.num_columns = great.num_cols
        self.fake_row = None

        self.prepare_start_prompt()

    def get_start_words(self, n_samples: int) -> tp.List[tp.List[int]]:
        return self.great_start.get_start_words(n_samples)

    def prepare_start_prompt(self):
        temp_dataset = GReaTDataset.from_pandas(pd.DataFrame(columns=self.columns))
        self.fake_row = temp_dataset._data.fast_slice()

    def get_start_tokens(self, n_samples: int) -> tp.List[tp.List[int]]:
        """ Overwrites default behavior to get start_tokens, by leveraging the Dataset specific one, and using the
        provided prompter to generate the corresponding text.
        :param n_samples: Number of samples to generate
        :return:
        """
        start_words = self.great_start.get_start_words(n_samples)
        # Use last item / column by default

        start_text = [
                [f'{self.columns[-1]} is '] + list(self.value_mapper(s)) + [", "]
            for s in start_words]

        start_tokens = self.tokenizer.batch_encode_plus(start_text, return_tensors='pt', is_split_into_words=True, padding=True, max_length=400)[
            'input_ids']
        return start_tokens


def get_prompter(prompter_name: str, *args) -> GReatPrompter:
    lookup = {
        'GReaTPrompter': GReatPrompter,
        'ContextPrompter': ContextPrompter,
        'ContextReorderedPrompter': ContextReorderedPrompter
    }
    return lookup[prompter_name](*args)
