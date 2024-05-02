#!/usr/bin/env zsh

datasets=(
  loan
  king
)

# Training with idea Context prompter
#echo "Running idea experiments without noise"
#for dataset in $datasets; do
#  echo "Training $dataset structured (clean)"
#  python3 -m structured_main --miss none --data $dataset train &> logs/structured/clean/${dataset}.clean.train.out
#done


echo "Running generation experiments without noise"
for dataset in $datasets; do
  echo "Training $dataset structured (clean)"
  python3 -m structured_main --miss none --data $dataset generate &> logs/structured/clean/${dataset}.clean.gen.out
done
