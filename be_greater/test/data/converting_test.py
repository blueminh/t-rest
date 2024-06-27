import unittest

import pandas as pd
from parameterized import parameterized

from src.data import convert_dataframe


def precision_func(testcase_func, param_num, param):
    assert len(param_num) == 1
    return "%s_%s" %(
        testcase_func.__name__,
        parameterized.to_safe_name(f'precision_{param.args[0]}'),
    )


class TestDataFrameConversionPosNeg(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(
            [{
                'float': 123.456789,
                'int': 1234,
                'string': "hello world!"
            },
            {
                'float': -1234.56789,
                'int': -123,
                'string': "goodbye world!"
            }
            ])

    def test_dataframe_equal_length(self):
        conversion_map = {
            'float': 2
        }
        converted_df, *_ = convert_dataframe(self.df, conversion_map=conversion_map)

        for column in converted_df.columns:
            self.assertEquals(len(converted_df[column].map(len).unique()), 1)

    @parameterized.expand([[1], [2], [3], [4], [10]], precision_func)
    def test_dataframe_equal_length(self, precision):
        conversion_map = {
            'float': precision
        }
        converted_df, *_ = convert_dataframe(self.df, conversion_map=conversion_map)

        # Floating points are [-+][0-9]{4}\.[0-9]{precision}
        self.assertEquals(converted_df['float'].map(len).unique()[0], 6 + precision)
        # Ints are [-+][0-9]{4}
        self.assertEquals(converted_df['int'].map(len).unique()[0], 5)

    @parameterized.expand([[1], [2], [3], [4], [10]], precision_func)
    def test_dataframe_equal_length_absolute(self, precision):
        conversion_map = {
            'float': precision
        }
        self.df[['float', 'int']] = self.df[['float', 'int']].abs()
        converted_df, *_ = convert_dataframe(self.df, conversion_map=conversion_map)

        # Floating points are [0-9]{4}\.[0-9]{precision}
        self.assertEquals(converted_df['float'].map(len).unique()[0], 5 + precision)
        # Ints are [-+][0-9]{4}
        self.assertEquals(converted_df['int'].map(len).unique()[0], 4)