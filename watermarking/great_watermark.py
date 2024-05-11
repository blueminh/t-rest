import torch
from transformers import (LogitsProcessor, AutoTokenizer)
import numpy as np
class GreatWatermarkLogitProcessor(LogitsProcessor):

    def __init__(
            self,
            tokenizer: AutoTokenizer,
            device,
            vocab: list[int] = None,
            gamma: float = 0.25,
            delta: float = 2.0,
    ):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.gamma = gamma
        self.delta = delta
        self.device = device
        self.rng = torch.Generator(device=device)

    def __convert_tokens_to_text(self, tokens):
        text_data = [self.tokenizer.decode(t) for t in tokens]
        # Clean text
        text_data = [d.replace("<|endoftext|>", "") for d in text_data]
        text_data = [d.replace("\n", " ") for d in text_data]
        text_data = [d.replace("\r", "") for d in text_data]
        text_data = ''.join(text_data)
        return text_data

    def __identify_column_name(self, text_sequence):
        # Split the token sequence by ', ' to get individual tokens
        try:
            cols = text_sequence.split(', ')
            last_col = cols[-1]
            if 'is' in last_col:
                return last_col.split('is')[0].strip()
            return None
        except Exception as e:
            return None

    def get_greenlist_ids(self, text_seed) -> torch.LongTensor:
        """Seed rng based on local context width and use this information to generate ids on the green list."""
        seed = sum(self.tokenizer.encode(text_seed))
        # print("col is {}, seed is {}".format(text_seed, seed))
        self.rng.manual_seed(seed % (2 ** 64 - 1))
        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
        greenlist_ids = vocab_permutation[:greenlist_size]  # new
        return greenlist_ids

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, list_of_greenlists) -> torch.BoolTensor:
        # Cannot lose loop, greenlists might have different lengths
        green_tokens_mask = torch.zeros_like(scores, dtype=torch.bool)
        for b_idx, greenlist in enumerate(list_of_greenlists):
            if len(greenlist) > 0:
                green_tokens_mask[b_idx][greenlist] = True
        return green_tokens_mask

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # NOTE, it would be nice to get rid of this batch loop, but currently,
        # the seed and partition operations are not tensor/vectorized, thus
        # each sequence in the batch needs to be treated separately.
        list_of_greenlist_ids = [None for _ in input_ids]
        for b_idx, input_seq in enumerate(input_ids):
            text_data = self.__convert_tokens_to_text(input_seq)
            current_col = self.__identify_column_name(text_data)
            # IMPORTANT: only apply watermarks on column's value
            # Use the column name as seed for rng
            greenlist_ids = self.get_greenlist_ids(current_col) if current_col else []
            # Update the green_list_ids
            list_of_greenlist_ids[b_idx] = greenlist_ids

        green_list_mask = self._calc_greenlist_mask(scores, list_of_greenlist_ids)
        scores[green_list_mask] = scores[green_list_mask] + self.delta

        return scores


class GreatWatermarkDetector:
    def __init__(
            self,
            watermark_logit_processor,
    ):
        self.watermark_logit_processor = watermark_logit_processor
        self.tokenizer = watermark_logit_processor.tokenizer

    def _count_greenlist_tokens_per_column(self, col_name, col_values):
        greenlist_ids = self.watermark_logit_processor.get_greenlist_ids(col_name)

        tokens = np.concatenate([self.tokenizer.encode(" " + str(col_val)) for col_val in col_values])
        is_in_green = torch.isin(torch.LongTensor(tokens), greenlist_ids)
        count_tokens_in_green_list = torch.sum(is_in_green).item()
        print("{}: {} out of {}".format(col_name, count_tokens_in_green_list, len(tokens)))
        # convert text values to tokens

    def detect(self, df):
        columns = [col.strip() for col in df.columns.tolist()]
        for column_name in columns:
            self._count_greenlist_tokens_per_column(column_name, df[column_name])

    def print_with_color(self, df):

        columns = [col.strip() for col in df.columns.tolist()]
        column_greenlists = {}
        column_max_width = {}  # Dictionary to store maximum width for each column

        for col_name in columns:
            column_greenlists[col_name] = self.watermark_logit_processor.get_greenlist_ids(col_name)
            max_text_length = max(len(str(value)) for value in df[col_name])
            column_max_width[col_name] = max_text_length

        for row_index, row in df.iterrows():  # a single row
            print()
            for col_index, col_name in enumerate(columns): # for each value
                value = row[col_name]
                token_ids = self.tokenizer.encode(" " + str(value))
                for token_index, token in enumerate(token_ids): # for each token within a value

                    color = '\033[92m' if token in column_greenlists[col_name] else '\033[91m'
                    end_color = '\033[0m'
                    token_text = self.tokenizer.decode(token)
                    # Print colored token with padding
                    print(f"{color}{token_text}{end_color}", end="")

                padding_length = column_max_width[col_name] - len(str(value)) + 1
                print(" " * padding_length, end="")
