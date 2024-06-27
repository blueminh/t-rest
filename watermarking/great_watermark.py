import scipy
import torch
from math import sqrt
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

    """
    Return the number of tokens in a column and the number of greenlist tokens
    """

    def _count_greenlist_tokens_per_column(self, col_name, col_values):
        greenlist_ids = self.watermark_logit_processor.get_greenlist_ids(col_name)

        tokens = np.concatenate([self.tokenizer.encode(" " + str(col_val)) for col_val in col_values])
        is_in_green = torch.isin(torch.LongTensor(tokens), greenlist_ids)
        count_tokens_in_green_list = torch.sum(is_in_green).item()
        # print("{}: {} out of {}".format(col_name, count_tokens_in_green_list, len(tokens)))
        return len(tokens), count_tokens_in_green_list

    def _compute_z_score(self, observed_count, T, gamma):
        # count refers to number of green tokens, T is total number of tokens
        expected_count = gamma * T
        # expected_count = gamma
        numer = observed_count - expected_count
        denom = sqrt(expected_count * (1 - gamma))
        z = numer / denom
        return z

    def _compute_p_value(self, z):
        p_value = scipy.stats.norm.sf(z)
        return p_value

    def _count_token_in_row(self, row):
        count = 0
        for _, value in row.items():
            count += len(self.tokenizer.encode(" " + str(value)))
        return count

    def power(self, x, n):
        # Higher softmax temp == softer ~ lower spike
        # e_x = np.exp((x - np.max(x)) / T)  # Subtract max for numerical stability
        # return e_x / e_x.sum(axis=0)

        square_x = np.power(x, n)
        return square_x / square_x.sum(axis=0)

    def softmax(self, x, T=1.0):
        e_x = np.exp((x - np.mean(x)) / T)  # Subtract max for numerical stability
        return e_x / e_x.sum(axis=0)

    def _calculate_entropy(self, col):
        value_counts = col.value_counts()
        probabilities = value_counts / len(col)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def get_column_weights(self, df, softmax_temp=5.0, custom_softmax=None):
        entropies = {}
        for col in df.columns:
            entropies[col] = self._calculate_entropy(df[col])

        entropy_values = np.array(list(entropies.values()))
        if custom_softmax:
            softmax_weights = custom_softmax(entropy_values)
        else:
            softmax_weights = self.softmax(entropy_values, softmax_temp)
        # Map the softmax weights back to column names
        column_weights = dict(zip(entropies.keys(), softmax_weights * len(entropies.values())))

        return column_weights

    def detect(self, df, included_columns=None, total_tokens_limit=200, random_state=42, softmax_temperature=2.0, custom_softmax=None):
        if included_columns:
            df = df[included_columns]
        column_weights = self.get_column_weights(df, softmax_temp=softmax_temperature, custom_softmax=custom_softmax)
        # z-score increases as length increases. Even real data might yield a very high z-score
        # count the number of tokens in a row to estimate the number of rows to consider
        tokens_per_row = self._count_token_in_row(df.iloc[0])
        # if there are too many tokens, randomly sample a number of rows in the df
        num_row = total_tokens_limit // tokens_per_row
        df = df.sample(n=num_row, random_state=random_state)

        columns = [col.strip() for col in df.columns.tolist()]
        # calculate the score for each column
        col_stats = {}
        total_tokens = 0
        total_greenlist_tokens = 0
        # total_weighted_tokens = 0
        total_weighted_greenlist_tokens = 0
        for column_name in columns:
            col_total_tokens, col_greenlist_tokens = self._count_greenlist_tokens_per_column(column_name,
                                                                                             df[column_name])
            weighted_greenlist_tokens = col_greenlist_tokens * column_weights[column_name]
            col_stats[column_name] = {
                "total_tokens": col_total_tokens,
                "greenlist_tokens": col_greenlist_tokens,
                "greenlist_percentage": col_greenlist_tokens * 1.0 / col_total_tokens,
                "weighted_greenlist_tokens": weighted_greenlist_tokens
            }
            # calculate total tokens
            total_tokens += col_total_tokens
            # total_weighted_tokens += col_total_tokens * column_weights[column_name]

            # calculate green list tokens
            total_greenlist_tokens += col_greenlist_tokens
            total_weighted_greenlist_tokens += weighted_greenlist_tokens

        z_score = self._compute_z_score(total_greenlist_tokens, total_tokens, self.watermark_logit_processor.gamma)
        weighted_z_score = self._compute_z_score(total_weighted_greenlist_tokens, total_tokens, self.watermark_logit_processor.gamma)
        # weighted_z_score_adjusted_total = self._compute_z_score(total_weighted_greenlist_tokens, total_weighted_tokens, self.watermark_logit_processor.gamma)

        final_scores = {
            "z_score": z_score,
            "p_value": self._compute_p_value(z_score),
            "weighted_z_score": weighted_z_score,
            # "weighted_z_score_adjusted_total": weighted_z_score_adjusted_total,
            "total_tokens": total_tokens,
            # "total_weighted_tokens": total_weighted_tokens,
            "greenlist_tokens": total_greenlist_tokens,
            "weighted_greenlist_tokens": total_weighted_greenlist_tokens,
            "greenlist_percentage": total_greenlist_tokens * 1.0 / total_tokens,
            "num_row": num_row,
        }
        return final_scores

    def print_with_color(self, df, num_rows=50):

        columns = [col.strip() for col in df.columns.tolist()]
        column_greenlists = {}
        column_max_width = {}  # Dictionary to store maximum width for each column

        for col_name in columns:
            column_greenlists[col_name] = self.watermark_logit_processor.get_greenlist_ids(col_name)
            max_text_length = max(len(str(value)) for value in df[col_name])
            max_text_length = max(max_text_length, len(col_name))
            column_max_width[col_name] = max_text_length

        for col_name in columns:
            print(" " + col_name, end="")
            padding_length = column_max_width[col_name] - len(col_name) + 1
            print(" " * padding_length, end="")
        print()  # Print newline after printing column names

        for row_index, row in df.iterrows():  # a single row
            for col_index, col_name in enumerate(columns):  # for each value
                value = row[col_name]
                token_ids = self.tokenizer.encode(" " + str(value))
                for token_index, token in enumerate(token_ids):  # for each token within a value

                    color = '\033[92m' if token in column_greenlists[col_name] else '\033[91m'
                    # color = '\033[30m' # black
                    end_color = '\033[0m'
                    token_text = self.tokenizer.decode(token)
                    # Print colored token with padding
                    print(f"{color}{token_text}{end_color}", end="")

                padding_length = column_max_width[col_name] - len(str(value)) + 1
                print(" " * padding_length, end="")
            print()
            if row_index == num_rows:
                break
