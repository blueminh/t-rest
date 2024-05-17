import glob
import re
import os


def create_file_name(model_name, sample_size, is_watermarked, gamma=None, delta=None):
    if is_watermarked:
        return "{}_{}_{}_gamma-{}_delta-{}.csv".format(
            model_name,
            sample_size,
            "with-watermark",
            gamma,
            delta
        )
    return "{}_{}_{}.csv".format(model_name, sample_size, "non-watermark")
