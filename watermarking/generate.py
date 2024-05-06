from be_great import GReaT
import transformers

from lm_watermarking.extended_watermark_processor import WatermarkLogitsProcessor

model_path = "/Users/minhkau/Documents/TUDelft/Year 3/RP/Code/tabular-gpt/be_greater/models/adult_v_baseline"

# loading and setting up
great = GReaT.load_from_dir(model_path)
great.tokenizer = transformers.AutoTokenizer.from_pretrained('distilgpt2', add_prefix_space=True)
great.tokenizer.pad_token_id = great.tokenizer.eos_token_id
great.tokenizer.pad_token = great.tokenizer.eos_token

# set up watermark
watermark_processor = WatermarkLogitsProcessor(vocab=list(great.tokenizer.get_vocab().values()),
                                               gamma=0.25,
                                               delta=2.0,
                                               seeding_scheme="selfhash")

# generation
samples = great.sample(100, k=10, max_length=400, device="cpu", logits_processor=watermark_processor)
samples.to_csv("watermaked_adult_samples_100.csv", index=False)