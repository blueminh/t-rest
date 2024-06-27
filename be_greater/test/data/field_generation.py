import  unittest

import numpy as np
import torch
import transformers

from src.data.trie import FieldGuide, IntegerTrie, ContinuousTrie, TokenTrie


class TestCaseField(unittest.TestCase):

    def setUp(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("distilgpt2")

    def test_integer_trie_pos_neg(self):

        integer_trie = IntegerTrie(-10, 300)
        field_guide = FieldGuide(['This', 'Column'], [' is', ' '], integer_trie, [', '])

        cur = []
        while True:
            print((cur := cur +  field_guide.next(cur)[:1]))

    def test_float_trie_pos_neg(self):

        floating_trie = ContinuousTrie(-10, 300, 3, 3, tokenizer=self.tokenizer)
        field_guide = FieldGuide(['This', 'Column'], [' is', ' '], floating_trie, [', '])

        cur = ["This", 'Came', "Before", "But", "Should", "Not", "Matter"]
        while True:
            print((cur := cur + field_guide.next(np.array(cur))[:1]))

    def test_tokenTrie(self):

        token_trie = TokenTrie()
        token_trie.insert([1, 2, 3])
        token_trie.insert([1, 3, 5])
        token_trie.insert([2, 4, 6, 7])
        field_guide = FieldGuide(['This', 'Column'], [' is', ' '], token_trie, [', '])

        cur = []
        while True:
            next = field_guide.next(np.array(cur))[-1:]
            (cur := cur + next)
            print(cur)