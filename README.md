# mtr-en2th

Welcome to the "mtr-en2th" project!

This project is a machine translation system that translates English song lyrics into Thai using transformer architecture. The system employs the state-of-the-art deep learning techniques and tools to achieve the translation task.

The project uses the transformer architecture, which is a powerful neural network model that has significantly improved the performance of various natural language processing tasks. Additionally, we use SentencePiece for the tokenizer, which is a subword tokenization library that allows us to segment words into smaller subword units, resulting in better translation quality.

And to evaluate the quality of the translations, we use the Meteor metric, which is a popular evaluation metric for machine translation systems.

This project is the final project for our deep learning course, and we have implemented and trained the model from scratch. We hope that this project will showcase the potential of deep learning in natural language processing and inspire further research in this field.

We hope you enjoy exploring our project and the results of our translation system!

## Try Now!

You can try the machine translation model in action at the following [🤗 space](https://huggingface.co/spaces/napatswift/en2th/tree/main?logs=container)


## Getting Started
To get started with the mtr-en2th project, it is recommended to build a virtual environment using `venv`. This ensures a clean and isolated environment to work with and that all the necessary dependencies are installed correctly.

To create a virtual environment, run the following command:

```bash
python -m venv env
```

This will create a new directory called `env` that contains all the necessary files for your virtual environment.

Activate the virtual environment by running the following command:

```bash
source env/bin/activate
```

This will activate the virtual environment and allow you to install the required dependencies.

After activating the virtual environment, install the required dependencies using pip. Run the following command:

```bash
pip install -r requirements.txt
```

This command will install all the necessary dependencies for the mtr-en2th project.

### Data Preparation

The scb-mt-en-th-2020 dataset can be downloaded using the following command:

```bash
wget https://github.com/vistec-AI/thai2nmt/releases/download/scb-mt-en-th-2020_v1.0/en-th.merged_stratified.train.csv -P dataset
```

After obtaining the dataset, we run script as the following. This is optional since we already ran it for you. 

```
python dataset.py
```

We need to train two models: the tokenizer model and the language model.

### Tokenizer model

We use SentencePiece as our tokenizer since it is a language-independent and unsupervised learning model. To train the SentencePiece model, run the following script:

```bash
python sp-train.py
```

This script builds two models: one for English language and another for Thai language.


### Language model

The language model is the key component of our mtr-en2th project, responsible for translating English song lyrics into Thai. To achieve this task, we use the Transformer architecture, a powerful neural network model that has shown exceptional performance in various natural language processing tasks.

The Transformer architecture consists of two main components: the encoder and the decoder. The encoder take an input sequence of tokens and processes them to generate a series of embeddings. These hidden representations are passed to the decoder, which generates the output sequence of tokens. The decoder uses the encoder's hidden representations to attend to relevant information and generate the output sequence of tokens.

Our training script for the language model is included in this repository, which allows you to train the model from scratch. We provide various hyperparameters and settings that you can adjust to optimize the model's performance for your specific use case.


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
  --sentence_piece_eng_path 'spmodel/en.model' \
  --sentence_piece_tha_path 'spmodel/th.model'
```

By specifying these parameters, you can customize the behavior of our mtr-en2th program to suit your specific needs. Once you run this command, the program will start training the language model.


## Testing

To test the trained model, use the following command:

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

This command tests the trained model by translating English lyrics to Thai.

Here's the breakdown of the parameters used in this command:

- `model_path`: The path to the trained machine translation model that was saved during training.

- `spm_eng` and `spm_tha`: The paths to the trained SentencePiece models for English and Thai, respectively. These models are used to tokenize the input data before feeding it into the language model.

After running this command, the program will output the predicted Thai translation for each input English lyric in the test dataset. You can then use various metrics such as *METEOR* score to evaluate the quality of the translations.

# Contributions are welcomed