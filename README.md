# mtr-en2th

mtr-en2th is a project that uses transforms to perform machine translation of English song lyrics to Thai. This project can be useful for anyone who wants to understand the meaning of English songs in Thai, or for language learners who want to improve their English and Thai skills.

## Try Now!

We have a 🤗 space, you can try translate lyrics now at [this](https://huggingface.co/spaces/napatswift/en2th/tree/main?logs=container)

## Prepare data

download scb-mt-en-th-2020 data with this command
```
wget https://github.com/vistec-AI/thai2nmt/releases/download/scb-mt-en-th-2020_v1.0/en-th.merged_stratified.train.csv -P dataset
``` 

## Getting Started

To get started with mtr-en2th, you will need to build a virtual environment using venv. This will ensure that you have a clean and isolated environment to work with, and that all of the necessary dependencies are installed correctly.

To create a virtual environment, run the following command:

```bash
python -m venv env
```

This will create a new directory called env that contains all of the necessary files for your virtual environment.

Next, activate the virtual environment by running the following command:

```bash
source env/bin/activate
```
This will activate the virtual environment and allow you to install the required dependencies.

Once you have activated the virtual environment, you can install the required dependencies using pip. To do this, run the following command:

```bash
pip install -r requirements.txt
```

This will install all of the necessary dependencies for mtr-en2th.

## Running the Program

```
% python train.py -h
usage: train.py [-h] [--name NAME] [--max_sequence_length MAX_SEQUENCE_LENGTH]
                [--sentence_piece_eng_path SENTENCE_PIECE_ENG_PATH] [--sentence_piece_tha_path SENTENCE_PIECE_THA_PATH]
                [--epochs EPOCHS] [--batch_size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           Name of the machine translation model. Default is mtr-model.
  --max_sequence_length MAX_SEQUENCE_LENGTH
                        Maximum sequence length for input text. Default is 64.
  --sentence_piece_eng_path SENTENCE_PIECE_ENG_PATH
                        Path to the English sentencepiece model. Default is spmodel/en.model.
  --sentence_piece_tha_path SENTENCE_PIECE_THA_PATH
                        Path to the Thai sentencepiece model. Default is spmodel/th.model.
  --epochs EPOCHS       Number of epochs to train the machine translation model. Default is 10.
  --batch_size BATCH_SIZE
                        Batch size for training the machine translation model. Default is 1024.
```

To run the program, simply execute the following command:

```bash
python train.py \
  --name mtr-en2th \
  --epochs 100 \
  --batch_size 2048 \
  --max_sequence_length 64 \
  --sentence_piece_eng_path 'spmodel/english.model' \
  --sentence_piece_tha_path 'spmodel/thai.model'
```


## Testing

to test the trained model use the test script

```
% python test.py -h
usage: test.py [-h] --model_path MODEL_PATH --spm_eng SPM_ENG --spm_tha SPM_THA

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Path to the trained machine translation model
  --spm_eng SPM_ENG     Path to the English sentencepiece model
  --spm_tha SPM_THA     Path to the Thai sentencepiece model
```

here's the example

```bash
python test.py --model_path mtr-model \
  --spm_eng spmodel/en.model \
  --spm_tha spmodel/th.model
```