import unittest

import pandas as pd

from src.data import data_type
from src.data.structured_dataset import StructuredDataset
import transformers

class DatasetTest(unittest.TestCase):

    def test_dataset_generation(self):
        """Test method to verify that a row in a 'structured' dataframe is decoded (after encoding) to the same sentence.
        Returns:

        """

        df = pd.read_csv('../datasets/king.csv')
        df, precision_map = data_type.convert_dataframe(df, 'king')

        dataset = StructuredDataset.from_pandas(df, preserve_index=False)
        tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained('distilgpt2')
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.pad_token_id = 50256
        dataset.set_tokenizer(tokenizer)

        row = dataset._getitem(0)

        encoded = dataset.tokenizer.batch_encode_plus([
            row,
            [''.join(row)]
        ],
            return_tensors='pt', is_split_into_words=True, padding=True, max_length=405)['input_ids']
        # Remove up until the first eos_Token
        encoded_standard = dataset.tokenizer.decode(encoded[1, :(encoded == 50256).nonzero(as_tuple=True)[1][0]])
        encoded_proposed = dataset.tokenizer.decode(encoded[0])
        self.assertEquals(encoded_proposed, encoded_standard)