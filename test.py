import keras_nlp
import tensorflow as tf
import evaluate
import pandas as pd
import deepcut
import argparse
from tensorflow import keras
import numpy as np

MAX_SEQUENCE_LENGTH = 54

def decode_sequences(input_sentences):
    MAX_SEQUENCE_LENGTH = 64
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

if __name__ == '__main__':
    meteor = evaluate.load('meteor')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='Path to the trained machine translation model', type=str, required=True)
    parser.add_argument('--spm_eng', help='Path to the English sentencepiece model', type=str, required=True)
    parser.add_argument('--spm_tha', help='Path to the Thai sentencepiece model', type=str, required=True)
    configs = parser.parse_args()

    model = keras.models.load_model(configs.model_path)
    
    ds_df = pd.read_csv('dataset/translate_test.csv')

    eng_tokenizer = keras_nlp.tokenizers.SentencePieceTokenizer(
            configs.spm_eng)
    tha_tokenizer = keras_nlp.tokenizers.SentencePieceTokenizer(
            configs.spm_tha)

    test_eng_texts = ds_df['en'].astype(str).tolist()
    test_tha_texts = ds_df['th'].tolist()
    metric_scores = []
    batch_size = 1000
    for ix in range(int(len(test_eng_texts)//batch_size)):
        input_sentence = test_eng_texts[ix*batch_size:ix*batch_size+batch_size]
        print(input_sentence)
        translated = decode_sequences(tf.constant(input_sentence))
        translated_list = []
        for translated in translated.numpy():
          translated = translated.decode("utf-8")
          translated = (
              translated
              .replace("<pad>", "")
              .replace("<s>", "")
              .replace("</s>", "")
              .replace("⁇", "")
              .strip()
          )

          predictions = [" ".join(deepcut.tokenize(translated))]
          references = [" ".join(deepcut.tokenize(test_tha_texts[ix]))]
          results = meteor.compute(predictions=predictions, references=references)
          metric_scores.append(results['meteor'])
        print(round(results['meteor'], 2))
        print(np.mean(metric_scores))
    print(np.mean(metric_scores))
    