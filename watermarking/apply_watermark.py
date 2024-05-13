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
        sample_file_name = "{}_{}_{}.csv".format(model_name, sample_size, "with-watermark")
    else:
        samples = great.sample(sample_size, k=batch_size, max_length=400, device="cpu")
        sample_file_name = "{}_{}_{}_gamma-{}_delta-{}.csv".format(model_name, sample_size, "non-watermark", gamma,
                                                                   delta)

    if write_to_file:
        samples.to_csv(samples_dir + sample_file_name)


def detection(sample_name, model_name, gamma, delta, included_columns=None):
    samples_dir = "/Users/minhkau/Documents/TUDelft/Year 3/RP/Code/tabular-gpt/samples/"
    great = load_model(model_name)
    great_watermark_processor = GreatWatermarkLogitProcessor(tokenizer=great.tokenizer,
                                                             device="cpu",
                                                             vocab=list(great.tokenizer.get_vocab().values()),
                                                             gamma=gamma,
                                                             delta=delta)
    great_watermark_detector = GreatWatermarkDetector(great_watermark_processor)

    data = pd.read_csv(samples_dir + sample_name)
    final_scores = great_watermark_detector.detect(data, included_columns)

    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(final_scores)


detection("watermarked_180.csv",
          model_name="adult_v_baseline",
          gamma=0.24, delta=2.0,
          included_columns=["fnlwgt"])

# great_watermark_detector.detect(real_data)
