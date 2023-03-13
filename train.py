import pandas as pd
import random
import keras_nlp
import tensorflow as tf
import argparse


class Dataset:
    def __init__(self,
                 data_path,
                 max_seq_len,
                 sentence_piece_eng_path,
                 sentence_piece_tha_path) -> None:
        self.train_pairs = None
        self.val_pairs = None
        self.test_pairs = None
        self._build_dataset(data_path)

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

    def make_dataset(self, pairs, batch_size=128):
        eng_texts, tha_texts = zip(*pairs)
        eng_texts = list(eng_texts)
        tha_texts = list(tha_texts)
        dataset = tf.data.Dataset.from_tensor_slices((eng_texts, tha_texts))
        dataset = dataset.batch(batch_size)
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
        ds_df = pd.read_csv(csv_path)
        ds_df = ds_df[ds_df['Matched?'] & (
            ds_df.Thai != '<song title>') & (~ds_df.Thai.isna())]
        ds_df.loc[:, 'Thai'] = ds_df.Thai.astype(str)
        ds_df.loc[:, 'English'] = ds_df.English.astype(
            str).apply(lambda x: x.lower())
        text_pairs = ds_df[['English', 'Thai']].values.tolist()
        random.shuffle(text_pairs)
        num_val_samples = int(0.15 * len(text_pairs))
        num_train_samples = len(text_pairs) - 2 * num_val_samples
        self.train_pairs = text_pairs[:num_train_samples]
        self.val_pairs = text_pairs[num_train_samples:
                                    num_train_samples + num_val_samples]
        self.test_pairs = text_pairs[num_train_samples + num_val_samples:]


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
    parser.add_argument('--max_sequence_length', default=48, type=int)
    parser.add_argument('--sentence_piece_eng_path', default='spmodel/m48.model', type=str)
    parser.add_argument('--sentence_piece_tha_path', default='spmodel/m48.model', type=str)

    configs = parser.parse_args()
    dataset = Dataset('dataset/translate.csv', 
                      configs.max_sequence_length,
                      configs.sentence_piece_eng_path,
                      configs.sentence_piece_tha_path)
    model_builder = ModelBuilder()
    eng_vocab_size = dataset.eng_tokenizer.vocabulary_size()
    tha_vocab_size = dataset.tha_tokenizer.vocabulary_size()
    model = model_builder.build_model(eng_vocab_size, tha_vocab_size, configs.max_sequence_length)

    model.summary()
    model.compile(
        "rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(dataset.train, epochs=10, validation_data=dataset.val)

    model.save(configs.name)
