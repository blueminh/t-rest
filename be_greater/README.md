# Tabular Imputation

This repository contains a series of experiments and implementations for GPT like models to train on 
tabular data.


## Setup
You might need to fix some dependencies, but the package list is given in requirements.txt. Note that the `great` package
needs to be installed using `ssh`. So make sure to have added your SSH key to the gitlab instance at TUDelft.


Note, this is developed with python3.10, but newer versions likely work as well.
```bash
python3.10 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```


## Structured generation
Structured generation accounts for generating with constraints on the sampling process. This works as follows.


N.B. all the commands will assume you execute them from the projects' root directory, i.e. where you find the `README.md`.

### Pre-training

First, you will need to train the model. This is done using the following command:


```bash
python3 structured_main.py --data <DATASET> train
```

Where `<DATASET>` needs to be sub. with a dataset (lowercase) in the `./data` directory.

### Generation

After training, generation can be performed with the same script.


```bash
python3 structured_main.py --data <DATASET> generate \
  -modeldir PATH/TO/YOUR/MODEL/DIR \
  -outputpath PATH/TO/SAVE/GENERATED/CONTENT
```

Check that the paths above exist, and point to the correct relative / absolute directory.

As an example

```bash
python3 structured_main.py --data king train
python3 structured_main.py -d king --prompter GReaTPrompter generate -modeldir models/king_v_baseline -outpath ./results
```
If the progression bar is very slow, or does not move at all. This means that either:

1. The model has not trained enough, generally using 100 epochs should be around what you need.
2. You have used a model that was trained with a different type of configuration. You will need to set the configuration
    objects yourself, as we don't provide any checks.