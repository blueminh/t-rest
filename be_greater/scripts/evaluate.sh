#!/usr/bin/env zsh

datasets=(
  loan
  king
#  adult
)

## Training with idea Context prompter
#echo "Generating data with GReaT baseline"
for dataset in $datasets; do
  echo "Evaluating $dataset (clean)"
  python3 evaluate.py $dataset ContextPrompter --prediction predictions/v2/idea --mis none
  echo "Evaluating $dataset (horizonal)"
  python3 evaluate.py $dataset ContextPrompter --prediction predictions/v2/idea --mis hor
  echo "Evaluating $dataset  (vertical)"
  python3 evaluate.py $dataset ContextPrompter --prediction predictions/v2/idea --mis ver
done
