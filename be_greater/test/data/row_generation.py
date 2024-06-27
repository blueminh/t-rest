import  unittest

import pandas as pd
import torch
import transformers

from src.data import data_type
from src.data.structured_dataset import StructuredDataset, structured_dataset_from_df
from src.data.trie import FieldGuide, IntegerTrie, ContinuousTrie, TokenTrie, RowGuide


class TestCaseField(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained('distilgpt2')

    def test_row_generation_scheme(self):
        start_prompt = ['price', ' is', ' ', '0', '3', '1', '3', '0', '0', '0', ',', ' ']
        start_input_ids = [20888,   318,   220,    15,    18,    16,    18,    15,    15,    15,
            11,   220]
        dataset_name = 'king'
        df = pd.read_csv(f'../datasets/{dataset_name}.csv')

        df, df_modified, dataset, tokenizer, precision_map = structured_dataset_from_df(dataset_name, df, self.tokenizer)


        row_guide = RowGuide(df, dataset.tokenizer, precision_map=precision_map)
        row_guide.set_order(df.columns[:-1])
        row_guide.reset()
        count = 0
        # try:
        while (options := row_guide.next(torch.tensor(start_input_ids)))[0] not in (50256, None):
            count += 1
            # print(options, count)
            # print(''.join(tokenizer.decode(start_input_ids)))
            if isinstance(options[0], int) and options[0] < 10:
                options[0] = tokenizer.encode(str(options[0]))[0]
            elif isinstance(options[0], str):
                options[0] = tokenizer.encode(str(options[0]))[0]
            # print(len(options))
            start_input_ids += options[:1]
            # print(''.join(tokenizer.batch_decode([start_input_ids])))
        row_guide.reset()
        start_input_ids = [20888,   318,   220,    15,    18,    16,    18,    15,    15,    15,
            11, 220]
        while (options := row_guide.next(torch.tensor(start_input_ids)))[0] not in (50256, None):
            count += 1
            # print(options, count)
            # print(''.join(tokenizer.decode(start_input_ids)))
            if isinstance(options[0], int) and options[0] < 10:
                options[0] = tokenizer.encode(str(options[0]))[0]
            elif isinstance(options[0], str):
                options[0] = tokenizer.encode(str(options[0]))[0]
            start_input_ids += options[:1]
        # except:
        print(''.join(tokenizer.batch_decode([start_input_ids])))

