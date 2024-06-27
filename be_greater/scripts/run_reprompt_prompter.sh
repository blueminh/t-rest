#!/usr/bin/env zsh

datasets=(
  loan
  king
  intrusion
  adult
  credit
)

# Training with idea Context prompter
echo "Running idea experiments with noise"

for dataset in $datasets; do
  echo "Training $dataset vertical"
  python3 main.py --idea --prompter ContextPrompter  --miss ver --data $dataset train &> logs/21-7/${dataset}.ver.out
  echo "Training $dataset horizontal"
  python3 main.py --idea --prompter ContextPrompter  --miss hor --data $dataset train &> logs/21-7/${dataset}.hor.out
done

#echo "Runninng baseline experiments"
#for dataset in $datasets; do
#  echo "Training $dataset vertical (clean baseline)"
#  python3 main.py --prompter ContextPrompter --miss none --data $dataset train &> logs/${dataset}.baseline-clean.ver.out
#done
#
#echo "Running baseline experiments with noise"
#for dataset in $datasets; do
#  echo "Training $dataset vertical (baseline)"
#  python3 main.py --prompter GReaTPrompter  --miss ver --data $dataset train &> logs/21-7/${dataset}.baseline-noisy.ver.out
#  echo "Training $dataset horizontal (baseline)"
#  python3 main.py --prompter GReaTPrompter  --miss hor --data $dataset train &> logs/21-7/${dataset}.baseline-noisy.hor.out
#done
#
#for dataset in $datasets; do
#  echo "Generating $dataset clean"
#  python3 main.py --idea --data ${dataset}.csv generate -modeldir models/baseline/clean/${dataset}_{}_baseline_clean -outpath predictions/intermediate/idea/${dataset}_{}.clean.csv
#  echo "Generating $dataset horizontal"
#  python3 main.py --idea --data ${dataset}.csv generate -modeldir models/idea/ContextPrompter/${dataset}_{}_baseline_complete_mr40_mc05 -outpath predictions/intermediate/idea/${dataset}_{}.hor.csv
#  echo "Training $dataset vertical"
#  python3 main.py --data ${dataset}.csv generate -modeldir models/idea/ContextPrompter/${dataset}_{}_baseline_complete_mr40 -outpath predictions/intermediate/idea/${dataset}_{}.ver.csv
#done

#
