import abc
import enum
from abc import ABC
from itertools import chain, islice

import numpy as np
import pandas as pd
import torch
import transformers

import typing as tp
from pandas.api.types import is_float_dtype, is_integer_dtype

from src.data.data_type import format_int, format_float


@enum.unique
class GenState(enum.Enum):
    BOS = 'bos' # Begin of sentence
    FNG = 'fng' # Field Name Generation
    EOS = 'eos' # End of Sentence


class Trie(abc.ABC):
    """Abstract interface for basic Trie."""

    @abc.abstractmethod
    def insert(self):
        pass

    @abc.abstractmethod
    def get(self, path: tp.List[int], max_depth: int) -> object:
        pass

    @property
    @abc.abstractmethod
    def depth(self):
        pass

class TokenTrie(Trie):
    """Token Trie, allowing to generate a Trie based on tokenized samples."""

    def __init__(self, item = None):
        self.children: tp.Dict[tp.T, TokenTrie] = None
        self.item = item

    @property
    def is_root(self):
        return self.item == None

    @property
    def is_leaf(self):
        return not self.is_root and len(self.children) == 0
    @property
    def depth(self):
        if self.is_root:
            depth = 0
            children = list(self.children.values())
            while True:
                if len(children) == 0:
                    return depth
                # TRIE is balanced due to encoding
                child = children[0]
                children = child.children
        else:
            raise Exception("Only root node knows Trie depth")


    def insert(self, tokens: tp.List[int]):
        if self.children is None:
            self.children = dict()
        if len(tokens) > 0:
            [head, *tail] = tokens
            child = self.children.get(head, TokenTrie(head))
            self.children[head] = child
            if len(tail) > 0:
                child.insert(tail)

    def get(self, path: tp.List[int], max_depth: int) -> tp.Dict[int, Trie]:
        if path and len(path) > 0:
            # If we are traversing the path

            if self.children is None:
                return None
            [head, *tail] = path
            child = self.children.get(head, None)
            return child.get(tail, max_depth-1)
        else:
            return self.children



class NumberTrie(Trie, ABC):
    """ Abstract Trie for Constrained number generation, this provides an (abstract) baseclass for virtual Tries to
    allow the generation of numbers within a range, without requiring O(n^k) space complexity, for a range that spans
    k digits.

    Effectively, this trie emulates a complete Trie using a statemachine. Note, this is implemented with a bunch of
        if statements. So can be improved.


    Attributes:
        min: Minimum number that is allowed to be generated.
        min_array:
    """

    def __init__(self, min, min_array, min_char, max, max_array, max_char, number_map):
        self.min = min
        self.min_array = min_array
        self.min_char = min_char

        self.max = max
        self.max_array = max_array
        self.max_char = max_char
        self.number_map = number_map

        # Min and max flag. To track whether we are within the permitted range (below max, and above min)
        self.min_flag = False
        self.max_flag = False

    def bounded_number_constraint(self, len_gen):
        if self.max > 0:
            # In the case that we have numbers that exceed zero,
            if self.max_flag and self.min_flag:
                # any character is fair game, as we are within bounds.
                return self.number_map
            elif self.max_flag and not self.min_flag:
                # return anything above minimum
                return dict(islice(self.number_map.items(), self.min_array[len_gen], 10))
            elif not self.max_flag and self.min_flag:
                # return anything below maximum
                return dict(islice(self.number_map.items(), 0, self.max_array[len_gen]))
            else:
                if len_gen == len(self.min_char) - 1 and len(self.min_char) > 1:
                    # In case we are not yet 'bounded', we must be able ot
                    return dict(islice(self.number_map.items(), self.min_array[len_gen], 10))
                else:
                    # return anything between min and max
                    low, high = self.min_array[len_gen], self.max_array[len_gen]
                    if low > high:
                        high, low = low, high
                    return dict(islice(self.number_map.items(), low, high))

        else:
            if self.max_flag and self.min_flag:
                # any charcter is fair game, as we are within bounds.
                return self.number_map
            elif self.max_flag and not self.min_flag:
                # return anything above minimum
                return dict(islice(self.number_map.items(), 0, self.min_array[len_gen]))
            elif not self.max_flag and self.min_flag:
                # return anything below maximum
                return dict(islice(self.number_map.items(), self.max_array[len_gen], 10))
            else:
                # I.e. for number that start with a certain 'leading' numbers, e.g. specific latitudes or longitudes
                if len_gen == len(self.min_char) - 1 and len(self.min_char) > 1:
                    # In case we are not yet 'bounded', we must be able ot
                    return dict(islice(self.number_map.items(), self.max_array[len_gen], 10))
                else:
                    # max is a 'smaller' negative number
                    return dict(islice(self.number_map.items(), self.max_array[len_gen], self.min_array[len_gen] ))


def decimals_to_input_ids(
        digits: tp.List[int],
        tokenizer: transformers.AutoTokenizer
) -> tp.List[int]:
    """Helper function to convert a list of integers to input ids'. N.B. this method assumes that the base of the
     given samples is less or equal to 10.

    TODO: As an alternative, the vocabulary of the tokenizer can also be matched with a regex to find tokens that
        correspond completely to digits.
    Args:
        digits: List of integers.
        tokenizer:

    Returns:
        Input ID's corresponding to passed digits.
    """
    digit_representations = []
    for digit in digits:
        # TODO: For GPT2 like models, it is better I think to use " {digit}" instead of "{digit}". There are some papers
        #   on this on the encoding of numbers when passed to a GPT2 like model.
        digit_representations.append(f"{digit}")

    return tokenizer.batch_encode_plus(digit_representations)['input_ids']


class IntegerTrie(NumberTrie):
    """ Virtual token Trie based on number generation. Prevents the need to build a rapidely growing full-trie.

    (0) - 0
          1
          2
          ...
        + 0
          ...

    TODO: This method can be extended for GPT2 like models, to also account for all tokens tokens that represent
        numbers.
    """

    @property
    def depth(self):
        return 1 + self.digits

    def __init__(self, min, max, digits = None, tokenizer = None):
        """

        """
        self.plus = '+'
        self.minus = '-'
        number_representation = list(range(10))
        if tokenizer is not None:
            self.plus = tokenizer.encode(self.plus)[0]
            self.minus = tokenizer.encode(self.minus)[0]
            number_representation = decimals_to_input_ids(number_representation, tokenizer)
            number_representation = list(chain(*number_representation))
        number_map = {number: self for number in number_representation}

        if digits is None:
            digits = int(np.log10(np.max([abs(min), abs(max)])) + 1)
        self.digits = digits

        # Index of the maximum number
        min_format = format_int(min, True, self.digits)
        min_array = [int(char) for char in min_format[1:]]
        min_char = list(chain(*tokenizer.batch_encode_plus(list(min_format))['input_ids']))

        # Include the number itself
        max_format = format_int(max, True, self.digits)
        max_array = [int(char)+1 for char in max_format[1:]]
        max_char = list(chain(*tokenizer.batch_encode_plus(list(max_format))['input_ids']))
        super(IntegerTrie, self).__init__(min, min_array, min_char, max, max_array, max_char, number_map)

    def insert(self, *args, **kwargs):
        pass


    def get(self, path: tp.Union[tp.List[int], torch.LongTensor], max_depth: int) -> tp.Dict[int, Trie]:
        if len(path) <= 0:
            # 1. Generate Sign
            # Case 1, the prefix (+/- sign) must be added for generating the number
            options = dict()
            if self.min < 0 and self.max >= 0:
                # Only make plus and minus accessible if values can be above range
                options[self.plus] = self
                options[self.minus] = self
            elif self.min < 0 and self.max < 0:
                options[self.minus] = self
            elif self.min >= 0 and self.max >= 0:
                options[self.plus] = self
            return options

        else:
            # 2. Generate number
            # Otherwise the options depend on the number of samples that are available
            gen, len_gen = path[-1], len(path) - 1
            if len_gen >= self.digits:
                # Base case.
                # We don't return anything if we have already computed the entire number
                return None

            # Update flags

            if not self.max_flag and path.tolist() < self.max_char[:len(path)]:
                self.max_flag = True
            if not self.min_flag and path.tolist() > self.min_char[:len(path)]:
                self.min_flag = True

            ret = self.bounded_number_constraint(len_gen)
            return ret




class ContinuousTrie(NumberTrie):
    """ Virtual token Trie based on number generation. Prevents the need to build a rapidely growing complete trie.
    This method allows to enforce to create a Trie that generates continuous numbers with a fixed layout. E.g. rather
    than allowing a BPEncoding scheme to be used for 123.0223, it will be generated as `'1','2','3','.','0','2','2','3'
    ']'.

    (0) - 0
          1
          2
          ...
        + 0
          ...
    """

    @property
    def depth(self):
        # +/- decimals . fractionals
        return 2 + self.decimal_precision + self.fractional_precision

    def __init__(self, min, max, decimal_precision, fractional_precision, tokenizer=None):
        """

        """
        self.min = min
        self.max = max
        self.decimal_precision = decimal_precision
        self.fractional_precision = fractional_precision
        self.negative = self.min < 0


        self.plus = '+'
        self.minus = '-'
        self.period = '.'
        number_representation = list(range(10))
        if tokenizer is not None:
            self.plus = tokenizer.encode(self.plus)[0]
            self.minus = tokenizer.encode(self.minus)[0]
            self.period = tokenizer.encode(self.period)[0]

            number_representation = tokenizer.batch_encode_plus(list(map(str, number_representation)))['input_ids']
            number_representation = list(chain(*number_representation))
        number_map = {number: self for number in number_representation}

        # Index of the maximum number
        min_format = format_float(min, self.fractional_precision, True, self.decimal_precision)
        min_array = [int(char)+(max < 0) if char != '.' else -1 for char in min_format[1:]]
        min_char = list(chain(*tokenizer.batch_encode_plus(list(min_format))['input_ids']))

        # Include the number itself
        max_format = format_float(max, self.fractional_precision, True, self.decimal_precision)
        max_array = [int(char)+(max > 0) if char != '.' else -1 for char in max_format[1:]]
        max_char = list(chain(*tokenizer.batch_encode_plus(list(max_format))['input_ids']))
        super(ContinuousTrie, self).__init__(min, min_array, min_char, max, max_array, max_char, number_map)

    def insert(self, *args, **kwargs):
        pass

    def get(self, path: torch.LongTensor, max_depth: int) -> tp.Dict[int, Trie]:
        """Getter for floating point numnber
        Args:
            path:
            max_depth:

        Returns:

        """

        if len(path) >= self.fractional_precision + self.decimal_precision + 2:
            # Base case
            # No more continuous number can be generated after this.
            return None
        options = dict()
        if len(path) <= 0:
            # Case 1, the prefix (+/- sign) must be added for generating the number
            # if self.min < 0:
            options = dict()
            if self.min < 0 and self.max >= 0:
                # Only make plus and minus accessible if values can be above range
                options[self.plus] = self
                options[self.minus] = self
            elif self.min < 0 and self.max < 0:
                options[self.minus] = self
            elif self.min >= 0 and self.max >= 0:
                options[self.plus] = self
            return options
        else:
            if len(path) == (self.decimal_precision + 1):
                options[self.period] = self
                return options

            gen, len_gen = path[-1], len(path) - (1)


            if self.max > 0:
                if not self.max_flag and path.tolist() > self.max_char[:len(path)]:
                    self.max_flag = True
                if not self.min_flag and path.tolist() < self.min_char[:len(path)]:
                    self.min_flag = True
            else:
                # In case only negative numbers, this is switched.
                if not self.max_flag and path.tolist() < self.max_char[:len(path)]:
                    self.max_flag = True
                if not self.min_flag and path.tolist() > self.min_char[:len(path)]:
                    self.min_flag = True
            # 2. Generate number
            # Otherwise the options depend on the number of samples that are available
            # Update flags

            return self.bounded_number_constraint(len_gen)


def get_tries(
        df: pd.DataFrame,
        tokenizer: transformers.PreTrainedTokenizer,
        precision_map: tp.Dict[str, int]) -> tp.Tuple[str, tp.Dict[str, tp.List[int]], tp.Dict[str, Trie]]:
        """    Helper function to construct a forest of Tries for fields in a dataframe.
        Args:
            df: Original dataframe from which to construct a Trie.
            tokenizer: Tokenizer used during training and generation.
            precision_map: Mapping for precision
        Returns:

        """
        keys = df.columns
        field_tries = {}
        field_begins = {key: tokenizer.encode(key) for key in keys}
        for key in keys:
            key_dtype = df.dtypes[key]
            if is_float_dtype(key_dtype):
                # Determine precision
                min_value, max_value = df[key].min(),  df[key].max()
                # TODO: Retrieve field information
                extrema = df[key].abs().max()
                leading_numbers = int(np.log10(extrema) + 1)
                # Instantiate trie
                trie = ContinuousTrie(min_value, max_value, leading_numbers, precision_map[key], tokenizer=tokenizer)
            elif is_integer_dtype(key_dtype):
                # Determine precision
                min_value, max_value = df[key].min(),  df[key].max()
                extrema = df[key].abs().max()
                # Get base 10 encoding
                # +1 for +/- sign
                number_of_numbers = int(np.log10(extrema) + 1)
                trie = IntegerTrie(min_value, max_value, number_of_numbers, tokenizer=tokenizer)
            else:
                # String type, limit generation to unique string values
                trie = TokenTrie(None)
                for value in df[key].unique():
                    trie.insert(tokenizer.encode(str(value)))
            field_tries[key] = trie
        return keys, field_begins, field_tries

class AbstractFieldGuide(abc.ABC):
    """Interface to implement a field guide class for guided generation of a field or colum name and corresponding
        values.
    """

    @abc.abstractmethod
    def reset(self):
        """Reset the field guide in case it needs to be re-used, or during re-generation of a part of a sequence."""
        pass

    @abc.abstractmethod
    def next(self, generated):
        """Function to get next valid tokens from a FieldGuide.
        Args:
            generated: Thusfar generated tokens by the model.

        Returns:

        """
        pass


class FieldGuide(AbstractFieldGuide):
    """Concrete implmenetation of Field Guide to generate a column name and corresponding allowable values.
    """

    def __init__(self, field_begin, equation, value_trie: Trie, seperation=None, abs_position = None):
        """

        Args:
            field_begin: Field name to begin with
            equation: Equaltion (e.g. ' is ', or ': ')
            value_trie: Trie to generate value
            seperation: Field seperation
            abs_position:
        """
        if seperation is None:
            seperation = list()
        # Relative position in the generation process
        self.position = 0
        # Field name tokens
        self.field_begin = field_begin
        self.len_field_begin = len(field_begin)
        # Equation tokens
        self.equation = equation
        # Trie representing the values that can be generated.
        self.value_trie = value_trie
        # (Optional) attribute seperator
        self._seperation = seperation
        # Absolute position within generated text, i.e., the position after which the field begins.
        self.abs_position = abs_position
        # End of field flag (name, equation, value, seperator), where all parts but value are optional.
        self.eof = False
        self.reset()

    def reset(self):
        """Helper function to reset Field guide after generation.

        Returns:
            None
        """
        self.position = 0
        self.eof = False
        self.seperation = [*self._seperation]

    def next(self, generated):
        ret = None
        if self.abs_position is None:
            self.abs_position = generated.shape[-1] #  len(generated)
        if self.eof:
            # 0. If completed generation of text, then keep returning None.
            pass
        elif self.position < (self.len_field_begin):
            # 1. Generate field name
            ret = [self.field_begin[self.position]]
        elif self.position < (field_eqn_length := self.len_field_begin + len(self.equation)):
            # 2. Generate
            ret = [self.equation[self.position - self.len_field_begin]]
        elif (allowed_tokens := self.value_trie.get(generated[self.abs_position + field_eqn_length:], -1)) is not None:
            #              a123456789...
            # Generated is 'field_name is [TOKEN]*'
            # so generated content is generated[-]
            # generated = [abs_position_length] + [field_name_length] + ['equation_length'] + [...]
            # We only care about the last part to obtain
            # TODO: As optimization, we can keep track of the child that were are currently generating with, so
            # to prevent the need to perform 1 + 2 + ... + n = O(n^2) lookups during generation.
            ret = list(allowed_tokens.keys())
        else:
            ret = self.seperation[:1]
            if len(self.seperation) == 1:
                self.eof = True
            else:
                self.seperation = self.seperation[1:]
        self.position += 1
        return ret


class AbstractRowGuide(abc.ABC):
    """Interface to implement guided generation of a row of samples."""

    @abc.abstractmethod
    def get_field_guide(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def set_order(self, order: tp.List[str]):
        pass

    @abc.abstractmethod
    def next(self, generated_tokens: tp.Union[tp.List[int], torch.Tensor]) -> tp.Union[tp.List[int], torch.Tensor]:
        pass


class RowGuide(AbstractRowGuide):
    """Concrete
    """

    def __init__(self, df: pd.DataFrame, tokenizer: transformers.PreTrainedTokenizer, overhead = 3, precision_map = None,
                 equation = [' is', ' '], seperation = [',', ' ']):
        self.df = df
        keys, field_begins, field_tries = get_tries(df, tokenizer, precision_map)
        self.keys = keys
        self.field_begins = field_begins
        self.field_tries = field_tries

        # self.overhead = overhead
        self.state = GenState.BOS

        # TODO: make this nicer
        #   Right now we assume the order is everything except the last column (target column implicitly).
        self._order = df.columns[:-1]
        self.order = None
        self.field_guide = None
        self.tokenizer = tokenizer

        equation_representation = tokenizer.batch_encode_plus(list(map(str, equation)))['input_ids']

        equation_representation = list(chain(*equation_representation))
        seperation_representation = tokenizer.batch_encode_plus(list(map(str, seperation)))['input_ids']

        seperation_representation = list(chain(*seperation_representation))
        self.equation = equation_representation
        self.seperation = seperation_representation
        self.set_order(self.df.columns[:-1])
        self.reset()

    def get_field_guide(self):
        key = self.order[0]
        return FieldGuide(
                self.field_begins[key],
                self.equation,
                self.field_tries[key],
                self.seperation if len(self.order) > 1 else None
        )

    def reset(self):
        """Reset the RowGuide for an entry in a batch. It assumes that the first field, including seperator for the next
        field is already provided by the callee.
        Returns:
            Mone

        """
        self.state = GenState.FNG
        self.order = self._order.copy()
        self.field_guide = self.get_field_guide()

    def set_order(self, order: tp.List[str]):
        """Set generation order for row guide.
        Args:
            order: Order in which attributes are calculated.

        Returns:

        """
        self._order = order

    def next(self, generated_tokens: tp.Union[tp.List[int], torch.Tensor]) -> tp.Union[tp.List[int], torch.Tensor]:
        """Implements iteration through 'statemachine', leverages FieldGuide to actually generate the samples.

        """
        next_tokens = self.field_guide.next(generated_tokens)
        if next_tokens is None or len(next_tokens) == 0:
            self.order = self.order[1:]
            if len(self.order) == 0:
                self.state = GenState.EOS
                return [50256]
            else:
                self.field_guide = self.get_field_guide()
                return self.next(generated_tokens)

        return next_tokens


class StructuredFieldGuide(AbstractFieldGuide):

    def __init__(self, col_token, value_tokens, abs_position = None, tpe = None):
        """

        Args:
            field_begin: Field name to begin with
            equation: Equaltion (e.g. ' is ', or ': ')
            value_trie: Trie to generate value
            seperation: Field seperation
            abs_position:
        """
        self.tpe = tpe
        # Relative position in the generation process
        self.position = 0
        # Field name tokens
        self.field_begin = col_token
        self.value_tokens = value_tokens
        # Absolute position within the generative process
        self.abs_position = abs_position
        # End of field flag (name, equation, value, seperator), where all parts but value are optional.
        self.eof = False
        self.reset()

    def reset(self):
        """Helper function to reset Field guide after generation.

        Returns:
            None
        """
        self.position = 0
        self.eof = False

    def next(self, generated):
        ret = None
        if self.abs_position is None:
            self.abs_position = generated.shape[-1] #  len(generated)
        if self.eof:
            # 0. If completed generation of text, then keep returning None.
            pass
        elif self.position == 0:
            # 1. Generate field name
            ret = [self.field_begin]
        elif self.position == 1:
            ret = self.value_tokens
            self.eof = True
        self.position += 1
        return ret
