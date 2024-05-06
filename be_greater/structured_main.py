import argparse
import logging
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import transformers

from be_great import GReaT
from src.data.data_type import format_int, format_float, get_precision
from src.data.permuted_dataset import (StructuredGreatSampler, PromptedGReaTDataset, get_prompter,
                                       CompoundedGreatSampler)
from src.data.structured_dataset import structured_dataset_from_df
from src.data.trie import RowGuide
from src.greater.great import sample


# data_path = './datasets/credit.csv'
MODEL_BASE_DIR = Path('./models')
DATA_BASE_DIR = Path('./data')
DATASET_BASE_DIR = Path('./datasets')
OUTPUT_DIR = Path('./predictions')


def train(data_path: Path, model='distilgpt2', prompter_name: str = 'GReaTPrompter', structured=False):
    logger = logging.getLogger()
    logger.info(f"Loading dataset from path: {data_path}")
    df = pd.read_csv(data_path)
    dataset_name = str(data_path.name).split('.')[0]
    if not structured:

        great = GReaT(llm=model,  # Name of the large language model used (see HuggingFace for more options)
                      epochs=100,  # Number of epochs to train
                      save_steps=50000,  # Save model weights every x steps
                      logging_steps=10000,  # Log the loss and learning rate every x steps
                      experiment_dir=str(MODEL_BASE_DIR / 'structured' / 'clean' /
                                         f'trainer_{dataset_name}_structured'),
                      batch_size=32,  # Batch Size
                      )

        # Generate a Prompted dataset from the Dataframes.
        dataset: PromptedGReaTDataset = PromptedGReaTDataset.from_pandas(df, preserve_index=False)
        prompter = get_prompter(prompter_name)
        dataset.set_prompter(prompter=prompter)
        logger.warning(f"Running Structured GReaT")

        logger.info(f"Starting fitting baseline: {dataset_name}")
        great.fit(df, column_names=None, great_ds=dataset)
        great.save(str(MODEL_BASE_DIR / (f"{dataset_name}_v_baseline")))
    else:
        great = GReaT(llm=model,  # Name of the large language model used (see HuggingFace for more options)
                      epochs=100,  # Number of epochs to train
                      save_steps=50000,  # Save model weights every x steps
                      logging_steps=10000,  # Log the loss and learning rate every x steps
                      experiment_dir=str(MODEL_BASE_DIR / 'structured' / 'clean' /
                                         f'trainer_{dataset_name}_structured'),
                      batch_size=32,  # Batch Size
                      )

        # Generate a Prompted dataset from the Dataframes.
        df, df_modified, structured_dataset, tokenizer, precision_map = structured_dataset_from_df(dataset_name, df,
                                                                                                     great.tokenizer)

        logger.warning(f"Running Structured GReaT")

        logger.info(f"Starting fitting structured: {dataset_name}")
        great.fit(df, column_names=None, great_ds=structured_dataset)
        great.save(str(MODEL_BASE_DIR / (f"{dataset_name}_structured")))


def load_model(path: Path):
    great = GReaT.load_from_dir(str(path))
    great.tokenizer = transformers.AutoTokenizer.from_pretrained('distilgpt2', add_prefix_space=True)
    great.tokenizer.pad_token_id = great.tokenizer.eos_token_id
    great.tokenizer.pad_token = great.tokenizer.eos_token
    return great


def generate(model: GReaT, data_path: str, out_path: Path = OUTPUT_DIR / 'default.csv', sample_bs=200,
             categorical=False, prompter_name: str = 'GReaTPrompter', structured = False):
    """

    Args:
        model (GReaT): Great model to patch and generate with.
        data_path (str): Path to original dataset
        out_path (Path): Path to the output file
        sample_bs (int): Number of samples to generate with each forward. Larger is faster, but may exceed available
            memory.
        categorical (False): Wether the dataset to be generated has a target column that is categorical.
    """
    logging.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    dataset: PromptedGReaTDataset = PromptedGReaTDataset.from_pandas(df, preserve_index=False)
    prompter = get_prompter(prompter_name)
    dataset.set_prompter(prompter=prompter)

    n_samples = len(df)
    logging.info(f"Generating {n_samples} samples.")

    # Overwrite the sampling method of the model with our custom implementation of the sampling method. This integrates
    # the RowGuides in the sampling method.
    # TODO: To make the implementation more 'huggingface' compatible. We can maybe adapt the code to be more in line
    #   of LogitsProcessor(List)
    if not structured:
        sampler = CompoundedGreatSampler(model.tokenizer, prompter=get_prompter(prompter_name ), great=model)
        samples = model.sampwle(n_samples, k=sample_bs, max_length=400, device="cpu")

        samples.to_csv("adult_samples_42k.csv", index=False)
    if structured:
        # Example code of how to generate with a structured sampler
        df, df_modified, structured_dataset, tokenizer, precision_map = structured_dataset_from_df(dataset_name, df,
                                                                                                     model.tokenizer)
        generate_row_guides(df, model, precision_map, sample_bs, tokenizer)

        # Set the instances' sampel modethod to the custom sampling method.
        sample_type = type(model.model.sample)
        model.model.sample = sample_type(sample, model.model)
        try:
            model.tokenizer.pad_token = model.tokenizer.eos_token
            model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
            if not categorical:
                target = df[df.columns[-1]]
                extrema = target.abs().max()

                negative = target.min() < 0.0

                def partial(func, tpe, *args):
                    l_func = func
                    def apply(value):
                        nonlocal l_func
                        return l_func(tpe(value), *args)

                    return apply


                if pd.api.types.is_integer_dtype(target.dtype):
                    decimals = int(np.ceil(np.log10(np.floor(extrema) + 1)))

                    mapper = partial(format_int, int, negative, decimals)
                else:
                    precision = get_precision(dataset_name)[df.columns[-1]]
                    decimals = int(np.ceil(np.log10(extrema)) + 1) + precision

                    mapper = partial(format_float, float, precision, negative, decimals)
                sampler = StructuredGreatSampler(model.tokenizer, model, mapper)

                samples = model.sample(n_samples, k=sample_bs, max_length=400, great_start=sampler, reordered_prompt=False)

                samples.to_csv(str(out_path), index=False)
        except:
            logging.fatal("Potentially the amount of video memory is not sufficient to perform inference with default"
                          "batch size. Maybelower batch-size to lower than default {sample_bs}")
            print(traceback.format_exc())


def generate_row_guides(data, model, precision_map, sample_bs, tokenizer, limit=True):
    """Helper method to instantiate the row-guide that steers generation. Note that this could also be implemented as a
    LogitsProcessor.

    Args:
        data:
        model:
        precision_map:
        sample_bs:
        tokenizer:
        limit: Flag to indicate whether the output should be limited to 'possible' values. I.e. only tokens that
        were seen in

    Returns:

    """
    model.model.generation_guides = [RowGuide(data, tokenizer, precision_map=precision_map) for _ in range(sample_bs)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for training and testing a model')
    parser.add_argument("-d", "--data", help='Dataset name to load for experiment/evaluation.')
    parser.add_argument('--prompter', help="Which prompter to use for a datset.",
                        choices=['GReaTPrompter', 'ContextReorderedPrompter', 'ContextPrompter'],
                        default='GReaTPrompter')
    subparsers = parser.add_subparsers(dest='action', help='Available actions')

    # Training subparser
    train_parser = subparsers.add_parser('train', help='Train the model')

    # Testing subparser
    test_parser = subparsers.add_parser('generate', help='Generate data with the fine-tunded model')
    test_parser.add_argument("-modeldir", help='Output directory to store output', default=str(MODEL_BASE_DIR))
    test_parser.add_argument("-outpath", help='Output directory to store output', default=OUTPUT_DIR )
    args = parser.parse_args()

    if args.action == 'train':
        print("Training with ContextPrompter")
        # Training is relatively similar to the original idea.
        train(DATASET_BASE_DIR / f"{args.data}.csv")

    elif args.action == 'generate':
        logging.info(f"Loading GReaT model: {args.modeldir}")
        logging.info(f"Starting GReaT generation: {args.outpath}")

        dataset_name = str(args.data).split('.')[0]
        great = load_model(str(args.modeldir))
        data_path = DATASET_BASE_DIR / 'structured' / f"{dataset_name}.csv"

        # Perpare model to generate and synthesize samples
        generate(model=great, data_path=DATASET_BASE_DIR / f"{args.data}.csv" ,
                 out_path=Path(str(f'{args.outpath}/{dataset_name}.structured.csv')))

# python structured_main.py -d adult generate -modeldir models/adult_v_baseline -outpath ./results