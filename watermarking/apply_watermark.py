from be_great import GReaT
import transformers
import pandas as pd
import pprint
from great_watermark import GreatWatermarkLogitProcessor, GreatWatermarkDetector
from utils import create_file_name


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
        sample_file_name = create_file_name(model_name, sample_size, True, gamma, delta)
    else:
        samples = great.sample(sample_size, k=batch_size, max_length=400, device="cpu")
        sample_file_name = create_file_name(model_name, sample_size, False)

    if write_to_file:
        samples.to_csv(samples_dir + sample_file_name, index=False)

    return samples


def detection(sample_name, gamma, delta, included_columns=None, sample_df=None, print_tokens=False, tokens_limit=1000):
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
        cropped_data = data[included_columns] if included_columns else data
        great_watermark_detector.print_with_color(cropped_data)

    return great_watermark_detector.detect(data, included_columns, total_tokens_limit=tokens_limit)


# model_names = ["california", "abalone", "adult", "diabetes"]
# for model_name in model_names:
#     print(model_name)
#     generation(
#         model_name=model_name,
#         sample_size=1000,
#         batch_size=100,
#         write_to_file=True,
#         with_watermark=True,
#         gamma=0.5,
#         delta=2.0
#     )

# generation(
#     model_name="abalone",
#     sample_size=1000,
#     batch_size=100,
#     write_to_file=True,
#     with_watermark=True,
#     gamma=0.5,
#     delta=1.0
# )

pp = pprint.PrettyPrinter(indent=2)
pp.pprint(detection("diabetes_1000_with-watermark_gamma-0.25_delta-2.0.csv",
                    gamma=0.25,
                    delta=2.0,
                    print_tokens=True,
                    tokens_limit=200,
                    included_columns=["Glucose", "BloodPressure", "Insulin", "BMI", "Age", "Outcome"]
                    )
          )

pp = pprint.PrettyPrinter(indent=2)
pp.pprint(detection("diabetes.csv",
                    gamma=0.25,
                    delta=2.0,
                    print_tokens=True,
                    included_columns=["Glucose", "BloodPressure", "Insulin", "BMI", "Age", "Outcome"]
                    # included_columns=['fnlwgt','sex','occupation','class','relationship']
                    )
          )

# pp.pprint(detection("attacked_samples/random_noise_0.01_diabetes_1000_with-watermark_gamma-0.25_delta-2.0.csv",
#                     gamma=0.25,
#                     delta=2.0,
#                     print_tokens=True,
#                     tokens_limit=200
#                     )
#           )

# pp.pprint(detection("adult_1000_with-watermark_gamma-0.5_delta-2.0.csv",
#                     gamma=0.5,
#                     delta=2.0,
#                     print_tokens=True,
#                     tokens_limit=500)
#           )

# pp.pprint(detection("california.csv",
#                     gamma=0.5,
#                     delta=2.0,
#                     print_tokens=False,
#                     tokens_limit=250,)
#           )
# pp.pprint(detection("california_1000_with-watermark_gamma-0.5_delta-2.0.csv",
#                     gamma=0.5,
#                     delta=2.0,
#                     print_tokens=False,
#                     tokens_limit=250)
#           )

#
# pp.pprint(detection("adult.csv",
#                     gamma=0.5,
#                     delta=2.0,
#                     print_tokens=False,
#                     tokens_limit=250)
#           )
#
# pp.pprint(detection("adult_1000_with-watermark_gamma-0.25_delta-2.0.csv",
#                     gamma=0.25,
#                     delta=2.0,
#                     print_tokens=True,
#                     tokens_limit=250,
#                     included_columns=['fnlwgt', 'sex', 'occupation', 'class', 'education-num'])
#           )

