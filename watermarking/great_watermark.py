import torch
from transformers import (LogitsProcessor, AutoTokenizer)
from lm_watermarking.extended_watermark_processor import (WatermarkBase)
from lm_watermarking import alternative_prf_schemes


class GreatWatermark(LogitsProcessor):

    def __init__(
            self,
            tokenizer: AutoTokenizer,
            vocab: list[int] = None,
            gamma: float = 0.25,
            delta: float = 2.0,
    ):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.gamma = gamma
        self.delta = delta

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

    def _get_greenlist_ids(self, text_seed) -> torch.LongTensor:
        """Seed rng based on local context width and use this information to generate ids on the green list."""
        seed = hash(text_seed)
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
        self.rng = torch.Generator(device=input_ids.device)
        list_of_greenlist_ids = [None for _ in input_ids]  # Greenlists could differ in length
        for b_idx, input_seq in enumerate(input_ids):
            text_data = self.__convert_tokens_to_text(input_seq)
            current_col = self.__identify_column_name(text_data)
            # IMPORTANT: only apply watermarks on column's value
            greenlist_ids = self._get_greenlist_ids(input_seq) if current_col else []
            list_of_greenlist_ids[b_idx] = greenlist_ids

        green_list_mask = self._calc_greenlist_mask(scores, list_of_greenlist_ids)
        scores[green_list_mask] = scores[green_list_mask] + self.delta

        return scores
