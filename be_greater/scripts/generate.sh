#!/usr/bin/env zsh

datasets=(
  loan
  king
  adult
)

## Training with idea Context prompter
#echo "Generating data with GReaT baseline"
for dataset in $datasets; do
  echo "Generating $dataset baseline (clean)"
  python3 main.py --prompter GReaTPrompter --data ${dataset}.csv generate -modeldir models/baseline/clean/${dataset}_GReaTPrompter_baseline_clean -outpath predictions/v2/baseline/${dataset}_GReaTPrompter.clean.csv
  echo "Generating $dataset baseline (horizonal)"
  python3 main.py --prompter GReaTPrompter --data ${dataset}.csv generate -modeldir models/baseline/noisy/${dataset}_GReaTPrompter_baseline_mr40_mc05 -outpath predictions/v2/baseline/${dataset}_{}.hor.csv
  echo "Generating $dataset baseline (vertical)"
  python3 main.py --prompter GReaTPrompter --data ${dataset}.csv generate -modeldir models/baseline/noisy/${dataset}_GReatPrompter_baseline_mr40 -outpath predictions/v2/baseline/${dataset}_{}.ver.csv
done


echo "Generating data with idea"
for dataset in $datasets; do

  # Generate data for ContextPrompter + Noise Aware training (complete)
  echo "Generating $dataset idea + noise aware training (horizonal)"
  python3 main.py --prompter ContextPrompter --idea --data ${dataset}.csv generate -modeldir models/idea/noisy/${dataset}_{}_baseline_complete_mr40_mc05 -outpath predictions/v2/idea/${dataset}_{}.hor.csv
  echo "Generating $dataset idea + noise aware training (vertical)"
  python3 main.py --prompter ContextPrompter --idea --data ${dataset}.csv generate -modeldir models/idea/noisy/${dataset}_{}_baseline_complete_mr40 -outpath predictions/v2/idea/${dataset}_{}.ver.csv
done

