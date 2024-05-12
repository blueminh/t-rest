from be_great import GReaT
import transformers
import pandas as pd

from lm_watermarking.extended_watermark_processor import WatermarkLogitsProcessor
from great_watermark import GreatWatermarkLogitProcessor, GreatWatermarkDetector

model_path = "/Users/minhkau/Documents/TUDelft/Year 3/RP/Code/tabular-gpt/be_greater/models/adult_v_baseline"

# loading and setting up
great = GReaT.load_from_dir(model_path)
great.tokenizer = transformers.AutoTokenizer.from_pretrained('distilgpt2', add_prefix_space=True)
great.tokenizer.pad_token_id = great.tokenizer.eos_token_id
great.tokenizer.pad_token = great.tokenizer.eos_token

# set up watermark
# watermark_processor = WatermarkLogitsProcessor(vocab=list(great.tokenizer.get_vocab().values()),
#                                                gamma=0.25,
#                                                delta=5.0,
#                                                seeding_scheme="selfhash")

# Generation
great_watermark_processor = GreatWatermarkLogitProcessor(tokenizer=great.tokenizer,
                                                         device="cpu",
                                                         vocab=list(great.tokenizer.get_vocab().values()),
                                                         gamma=0.25,
                                                         delta=5.0)

samples_dir = "/Users/minhkau/Documents/TUDelft/Year 3/RP/Code/tabular-gpt/samples"
# sample_name = "great_watermaked_adult_samples_gamma_5.csv"
sample_name= "great_watermaked_adult_samples_1000_high_delta.csv"

# samples = great.sample(1000, k=100, max_length=400, device="cpu", logits_processor=great_watermark_processor)
# samples.to_csv(samples_dir  + "/" + sample_name, index=False)

syn_data = pd.read_csv(samples_dir + "/" + sample_name)
great_watermark_detector = GreatWatermarkDetector(great_watermark_processor)
great_watermark_detector.detect(syn_data)
great_watermark_detector.print_with_color(syn_data)
#
# real_data = pd.read_csv(samples_dir + "/adult.csv")
# great_watermark_detector.detect(real_data)