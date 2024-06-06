from be_great import GReaT
import transformers
import pandas as pd
import pprint
from exp_watermark import GreatEXPWatermarkDetector, GreatEXPWatermarkLogitProcessor
from utils import create_file_name_exp


def load_model(model_name):
    model_folder_dir = "/Users/minhkau/Documents/TUDelft/Year 3/RP/Code/tabular-gpt/models/"
    great = GReaT.load_from_dir(model_folder_dir + model_name)
    great.tokenizer = transformers.AutoTokenizer.from_pretrained('distilgpt2', add_prefix_space=True)
    great.tokenizer.pad_token_id = great.tokenizer.eos_token_id
    great.tokenizer.pad_token = great.tokenizer.eos_token
    return great


def generation(model_name, sample_size, batch_size=50, write_to_file=False, with_watermark=False):
    samples_dir = "/Users/minhkau/Documents/TUDelft/Year 3/RP/Code/tabular-gpt/samples/"
    great = load_model(model_name)
    batch_size = min(batch_size, sample_size)
    if with_watermark:
        great_exp_watermark_processor = GreatEXPWatermarkLogitProcessor(tokenizer=great.tokenizer,
                                                                        device="cpu",
                                                                        vocab=list(great.tokenizer.get_vocab().values()))
        samples = great.sample(sample_size, k=batch_size, max_length=400, device="cpu", sampling_strategy='greedy',
                               logits_processor=great_exp_watermark_processor)
        sample_file_name = create_file_name_exp(model_name, sample_size, True)
    else:
        samples = great.sample(sample_size, k=batch_size, max_length=400, device="cpu")
        sample_file_name = create_file_name_exp(model_name, sample_size, False)

    if write_to_file:
        samples.to_csv(samples_dir + sample_file_name, index=False)
    return samples


def detection(sample_name, included_columns=None, sample_df=None, tokens_limit=200):
    samples_dir = "/Users/minhkau/Documents/TUDelft/Year 3/RP/Code/tabular-gpt/samples/"
    tokenizer = transformers.AutoTokenizer.from_pretrained('distilgpt2', add_prefix_space=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    great_exp_watermark_processor = GreatEXPWatermarkLogitProcessor(tokenizer=tokenizer,
                                                                    device="cpu",
                                                                    vocab=list(tokenizer.get_vocab().values()))
    great_exp_watermark_detector = GreatEXPWatermarkDetector(great_exp_watermark_processor)
    data = sample_df if sample_df else pd.read_csv(samples_dir + sample_name)
    return great_exp_watermark_detector.detect(data, included_columns, total_tokens_limit=tokens_limit)


generation(
    model_name="abalone",
    sample_size=1000,
    batch_size=100,
    write_to_file=True,
    with_watermark=True,
)

# print(detection("adult_1000_with-watermark_exp.csv"))
