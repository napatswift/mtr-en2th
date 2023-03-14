import keras_nlp
import tensorflow as tf
# import evaluate
import pandas as pd
import random
# import deepcut

import numpy as np
from tensorflow import keras
from tensorflow_text.tools.wordpiece_vocab import (
    bert_vocab_from_dataset as bert_vocab,
)

model = keras.models.load_model('mtr-model/')
meteor = evaluate.load('meteor')

BATCH_SIZE = 64
EPOCHS = 1  # This should be at least 10 for convergence
MAX_SEQUENCE_LENGTH = 64
ENG_VOCAB_SIZE = 15000
THA_VOCAB_SIZE = 15000

EMBED_DIM = 256
INTERMEDIATE_DIM = 2048
NUM_HEADS = 8

eng_tokenizer = keras_nlp.tokenizers.SentencePieceTokenizer(
            'spmodel/m48.model')
tha_tokenizer = keras_nlp.tokenizers.SentencePieceTokenizer(
            'spmodel/m48.model')

ds_df = pd.read_csv('dataset/translate.csv')
ds_df = ds_df[ds_df['Matched?'] & (ds_df.Thai != '<song title>') & (~ds_df.Thai.isna())]
ds_df.loc[:,'Thai'] = ds_df.Thai.astype(str)
ds_df.loc[:, 'English'] = ds_df.English.astype(str).apply(lambda x: x.lower())

text_pairs = ds_df[['English', 'Thai']].values.tolist()

random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]

def decode_sequences(input_sentences):
    batch_size = tf.shape(input_sentences)[0]

    # Tokenize the encoder input.
    encoder_input_tokens = eng_tokenizer(input_sentences).to_tensor(
        shape=(None, MAX_SEQUENCE_LENGTH)
    )

    # Define a function that outputs the next token's probability given the
    # input sequence.
    def token_probability_fn(decoder_input_tokens):
        return model([encoder_input_tokens, decoder_input_tokens])[:, -1, :]

    # Set the prompt to the "[START]" token.
    prompt = tf.fill((batch_size, 1), tha_tokenizer.token_to_id("<s>"))

    generated_tokens = keras_nlp.utils.top_p_search(
        token_probability_fn,
        prompt,
        p=0.1,
        max_length=40,
        end_token_id=tha_tokenizer.token_to_id("</s>"),
    )
    generated_sentences = tha_tokenizer.detokenize(generated_tokens)
    return generated_sentences


test_eng_texts = [pair[0] for pair in test_pairs]
test_tha_texts = [pair[1] for pair in test_pairs]
for i in range(12):
    ix = random.randint(0, len(test_eng_texts))
    input_sentence = test_eng_texts[ix]
    translated = decode_sequences(tf.constant([input_sentence]))
    translated = translated.numpy()[0].decode("utf-8")
    translated = (
        translated
        .replace("<pad>", "")
        .replace("<s>", "")
        .replace("</s>", "")
        .replace("⁇", "")
        .strip()
    )
    print(f"** Example {i} **")
    print(input_sentence)
    print(translated)
    print(test_tha_texts[ix])

    predictions = [" ".join(deepcut.tokenize(translated))]
    references = [" ".join(deepcut.tokenize(test_tha_texts[ix]))]
    results = meteor.compute(predictions=predictions, references=references)
    print(round(results['meteor'], 2))
    print()