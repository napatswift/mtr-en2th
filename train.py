import pandas as pd
import random
import keras_nlp
import tensorflow as tf
import argparse

class Dataset:
    def __init__(self,
                 max_seq_len,
                 sentence_piece_eng_path,
                 sentence_piece_tha_path,
                 batch_size) -> None:
        self.batch_size = batch_size
        self.train_pairs = self._build_dataset('dataset/translate_train.csv')
        self.val_pairs = self._build_dataset('dataset/translate_val.csv')
        self.test_pairs = self._build_dataset('dataset/translate_test.csv')

        self.eng_tokenizer = keras_nlp.tokenizers.SentencePieceTokenizer(
            sentence_piece_eng_path)

        self.tha_tokenizer = keras_nlp.tokenizers.SentencePieceTokenizer(
            sentence_piece_tha_path)

        self.eng_start_end_packer = keras_nlp.layers.StartEndPacker(
            sequence_length=max_seq_len,
            pad_value=self.eng_tokenizer.token_to_id("<p>"),)

        self.tha_start_end_packer = keras_nlp.layers.StartEndPacker(
            sequence_length=max_seq_len + 1,
            start_value=self.tha_tokenizer.token_to_id("<s>"),
            end_value=self.tha_tokenizer.token_to_id("</s>"),
            pad_value=self.tha_tokenizer.token_to_id("<p>"),
        )

        self.train = self.make_dataset(self.train_pairs)
        self.val = self.make_dataset(self.val_pairs)
        self.test = self.make_dataset(self.test_pairs)

    def make_dataset(self, pairs,):
        eng_texts, tha_texts = zip(*pairs)
        eng_texts = list(eng_texts)
        tha_texts = list(tha_texts)
        dataset = tf.data.Dataset.from_tensor_slices((eng_texts, tha_texts))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.map(self.preprocess_batch,
                              num_parallel_calls=tf.data.AUTOTUNE)
        return dataset.shuffle(2048).prefetch(16).cache()

    def preprocess_batch(self, eng, tha):
        eng = self.eng_tokenizer(eng)
        tha = self.tha_tokenizer(tha)

        eng = self.eng_start_end_packer(eng)
        tha = self.tha_start_end_packer(tha)

        return (
            {
                "encoder_inputs": eng,
                "decoder_inputs": tha[:, :-1],
            }, tha[:, 1:],
        )

    def _build_dataset(self, csv_path: str):
        ds_df = pd.read_csv(csv_path).dropna()
        return ds_df.values


class ModelBuilder:
    def __init__(self) -> None:
        pass

    def build_model(
            self,
            source_vocab_size,
            target_vocab_size,
            max_sequence_length,
            intermediate_dim=2048,
            embedding_dim=256,
            head_num=8,
    ):
        # Encoder
        encoder_inputs = tf.keras.Input(
            shape=(None,), dtype="int64", name="encoder_inputs")

        x = keras_nlp.layers.TokenAndPositionEmbedding(
            vocabulary_size=source_vocab_size,
            sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
            mask_zero=True,
        )(encoder_inputs)

        encoder_outputs = keras_nlp.layers.TransformerEncoder(
            intermediate_dim=intermediate_dim, num_heads=head_num
        )(inputs=x)
        encoder = tf.keras.Model(encoder_inputs, encoder_outputs)

        # Decoder
        decoder_inputs = tf.keras.Input(
            shape=(None,), dtype="int64", name="decoder_inputs")
        encoded_seq_inputs = tf.keras.Input(
            shape=(None, embedding_dim), name="decoder_state_inputs")

        x = keras_nlp.layers.TokenAndPositionEmbedding(
            vocabulary_size=target_vocab_size,
            sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
            mask_zero=True,
        )(decoder_inputs)

        x = keras_nlp.layers.TransformerDecoder(
            intermediate_dim=intermediate_dim, num_heads=head_num
        )(decoder_sequence=x, encoder_sequence=encoded_seq_inputs)
        x = tf.keras.layers.Dropout(0.5)(x)
        decoder_outputs = tf.keras.layers.Dense(
            target_vocab_size, activation="softmax")(x)
        decoder = tf.keras.Model([
            decoder_inputs,
            encoded_seq_inputs,
        ],
            decoder_outputs,
        )
        decoder_outputs = decoder([decoder_inputs, encoder_outputs])

        transformer = tf.keras.Model(
            [encoder_inputs, decoder_inputs],
            decoder_outputs,
            name="transformer",
        )

        return transformer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='mtr-model', type=str)
    parser.add_argument('--max_sequence_length', default=64, type=int)
    parser.add_argument('--sentence_piece_eng_path', default='spmodel/english.model', type=str)
    parser.add_argument('--sentence_piece_tha_path', default='spmodel/thai.model', type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)

    configs = parser.parse_args()
    dataset = Dataset(configs.max_sequence_length,
                      configs.sentence_piece_eng_path,
                      configs.sentence_piece_tha_path,
                      configs.batch_size)
    model_builder = ModelBuilder()
    eng_vocab_size = dataset.eng_tokenizer.vocabulary_size()
    tha_vocab_size = dataset.tha_tokenizer.vocabulary_size()
    model = model_builder.build_model(eng_vocab_size, tha_vocab_size, configs.max_sequence_length)

    model.summary()
    model.compile(
        "rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(dataset.train, epochs=configs.epochs, validation_data=dataset.val)

    model.save(configs.name)
