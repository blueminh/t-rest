from be_great import GReaT
import transformers
import pandas as pd
import pprint
from great_watermark import GreatWatermarkLogitProcessor, GreatWatermarkDetector


def load_model(model_name):
    model_folder_dir = "/Users/minhkau/Documents/TUDelft/Year 3/RP/Code/tabular-gpt/models/"
    great = GReaT.load_from_dir(model_folder_dir + model_name)
    great.tokenizer = transformers.AutoTokenizer.from_pretrained('distilgpt2', add_prefix_space=True)
    great.tokenizer.pad_token_id = great.tokenizer.eos_token_id
    great.tokenizer.pad_token = great.tokenizer.eos_token
    return great


def generation(model_name, sample_size, batch_size=50, write_to_file=False, with_watermark=False, gamma=0.25,
               delta=2.0):
    samples_dir = "/Users/minhkau/Documents/TUDelft/Year 3/RP/Code/tabular-gpt/samples/"
    great = load_model(model_name)
    batch_size = min(batch_size, sample_size)
    if with_watermark:
        great_watermark_processor = GreatWatermarkLogitProcessor(tokenizer=great.tokenizer,
                                                                 device="cpu",
                                                                 vocab=list(great.tokenizer.get_vocab().values()),
                                                                 gamma=gamma,
                                                                 delta=delta)
        samples = great.sample(sample_size, k=batch_size, max_length=400, device="cpu",
                               logits_processor=great_watermark_processor)
        sample_file_name = "{}_{}_{}_gamma-{}_delta-{}.csv".format(model_name, sample_size, "with-watermark", gamma,
                                                                   delta)
    else:
        samples = great.sample(sample_size, k=batch_size, max_length=400, device="cpu")
        sample_file_name = "{}_{}_{}.csv".format(model_name, sample_size, "non-watermark")

    if write_to_file:
        samples.to_csv(samples_dir + sample_file_name, index=False)

    return samples


def detection(sample_name, gamma, delta, included_columns=None, sample_df=None, print_tokens=False):
    samples_dir = "/Users/minhkau/Documents/TUDelft/Year 3/RP/Code/tabular-gpt/samples/"
    tokenizer = transformers.AutoTokenizer.from_pretrained('distilgpt2', add_prefix_space=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    great_watermark_processor = GreatWatermarkLogitProcessor(tokenizer=tokenizer,
                                                             device="cpu",
                                                             vocab=list(tokenizer.get_vocab().values()),
                                                             gamma=gamma,
                                                             delta=delta)
    great_watermark_detector = GreatWatermarkDetector(great_watermark_processor)
    data = sample_df if sample_df else pd.read_csv(samples_dir + sample_name)
    if print_tokens:
        great_watermark_detector.print_with_color(data)

    return great_watermark_detector.detect(data, included_columns)


# generation(
#     model_name="california",
#     sample_size=100,
#     batch_size=10,
#     write_to_file=True,
#     with_watermark=True,
#     gamma=0.25,
#     delta=2.0
# )

pp = pprint.PrettyPrinter(indent=2)
pp.pprint(detection("california_100_non-watermark.csv",
                    gamma=0.25,
                    delta=2.0,
                    print_tokens=True)['z_score'])

pp.pprint(detection("california_100_with-watermark_gamma-0.25_delta-2.0.csv",
                    gamma=0.25,
                    delta=2.0,
                    print_tokens=True)['z_score']
          )
