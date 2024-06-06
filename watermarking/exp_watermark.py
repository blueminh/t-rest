import scipy
import torch
from math import sqrt
from transformers import (LogitsProcessor, AutoTokenizer)
import numpy as np
import torch.nn.functional as F
import math


class GreatEXPWatermarkLogitProcessor(LogitsProcessor):

    def __init__(
            self,
            tokenizer: AutoTokenizer,
            device,
            vocab: list[int] = None,
    ):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.vocab_size = len(vocab)
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

    # this is the only required method
    def get_r(self, text_seed):
        seed = sum(self.tokenizer.encode(text_seed))
        # print("col is {}, seed is {}".format(text_seed, seed))
        self.rng.manual_seed(seed % (2 ** 64 - 1))
        r = torch.rand(self.vocab_size, generator=self.rng)
        return r

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for b_idx, input_seq in enumerate(input_ids):
            text_data = self.__convert_tokens_to_text(input_seq)
            current_col = self.__identify_column_name(text_data)
            if current_col:
                # gumbell softmax
                logits = scores[b_idx]
                probs = F.softmax(logits, dim=0)
                r = self.get_r(current_col)
                next_token = torch.argmax(torch.pow(r, 1 / probs))
                logits[:] = -math.inf
                logits[next_token] = 0
                scores[b_idx] = logits
        return scores


class GreatEXPWatermarkDetector:
    def __init__(
            self,
            watermark_logit_processor,
    ):
        self.watermark_logit_processor = watermark_logit_processor
        self.tokenizer = watermark_logit_processor.tokenizer

    def _count_token_in_row(self, row):
        count = 0
        for _, value in row.items():
            count += len(self.tokenizer.encode(" " + str(value)))
        return count

    def _get_r_score_per_column(self, col_name, col_values):
        sum = 0
        count_token = 0
        r = self.watermark_logit_processor.get_r(col_name)
        col_values = col_values
        for col_val in col_values:
            tokens_ids = self.tokenizer.encode(" " + str(col_val))
            count_token += len(tokens_ids)
            for token_id in tokens_ids:
                sum += np.log2(1.0 / (1 - r[token_id]))
        return count_token, sum

    def detect(self, df, included_columns=None, total_tokens_limit=200, random_state=42):
        if included_columns:
            df = df[included_columns]
        # count the number of tokens in a row to estimate the number of rows to consider
        tokens_per_row = self._count_token_in_row(df.iloc[0])
        # if there are too many tokens, randomly sample a number of rows in the df
        num_row = total_tokens_limit // tokens_per_row
        df = df.sample(n=num_row, random_state=random_state)

        columns = [col.strip() for col in df.columns.tolist()]

        # calculate the score for each column
        final_sum = 0
        total_tokens_count = 0
        for column_name in columns:
            count_tokens, sum_per_col = self._get_r_score_per_column(column_name, df[column_name])
            final_sum += sum_per_col
            total_tokens_count += count_tokens

        return {
            "total_tokens_count": total_tokens_count,
            "final_sum": final_sum.item()
        }
