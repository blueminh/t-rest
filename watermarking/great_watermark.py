import torch
from transformers import (LogitsProcessor, AutoTokenizer)


class GreatWatermark(LogitsProcessor):

    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        for input_index, input_seq in enumerate(input_ids):
            text_data = self.__convert_tokens_to_text(input_seq)
            current_col = self.__identify_column_name(text_data)
        return scores

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
