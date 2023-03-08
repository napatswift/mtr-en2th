import pandas as pd
import random
import keras_nlp
import tensorflow as tf

BATCH_SIZE = 64
EPOCHS = 1
MAX_SEQUENCE_LENGTH = 40
ENG_VOCAB_SIZE = 15000
THA_VOCAB_SIZE = 15000

EMBED_DIM = 256
INTERMEDIATE_DIM = 2048
NUM_HEADS = 8


class Dataset:
    def __init__(self) -> None:
        self.train_pairs = None
        self.val_pairs = None
        self.test_pairs = None
        self._build_dataset()

        self.eng_tokenizer = keras_nlp.tokenizers.SentencePieceTokenizer(
            'spmodel/m48.model')
        self.tha_tokenizer = keras_nlp.tokenizers.SentencePieceTokenizer(
            'spmodel/m48.model')

        self.eng_start_end_packer = keras_nlp.layers.StartEndPacker(
            sequence_length=MAX_SEQUENCE_LENGTH,
            pad_value=self.eng_tokenizer.token_to_id("<p>"),)

        self.tha_start_end_packer = keras_nlp.layers.StartEndPacker(
            sequence_length=MAX_SEQUENCE_LENGTH + 1,
            start_value=self.tha_tokenizer.token_to_id("<s>"),
            end_value=self.tha_tokenizer.token_to_id("</s>"),
            pad_value=self.tha_tokenizer.token_to_id("<p>"),
        )

        self.train = self.make_dataset(self.train_pairs)
        self.val = self.make_dataset(self.val_pairs)
        self.test = self.make_dataset(self.test_pairs)

    def make_dataset(self, pairs):
        eng_texts, tha_texts = zip(*pairs)
        eng_texts = list(eng_texts)
        tha_texts = list(tha_texts)
        dataset = tf.data.Dataset.from_tensor_slices((eng_texts, tha_texts))
        dataset = dataset.batch(BATCH_SIZE)
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

    def build_model(self):
        # Encoder
        encoder_inputs = tf.keras.Input(
            shape=(None,), dtype="int64", name="encoder_inputs")

        x = keras_nlp.layers.TokenAndPositionEmbedding(
            vocabulary_size=ENG_VOCAB_SIZE,
            sequence_length=MAX_SEQUENCE_LENGTH,
            embedding_dim=EMBED_DIM,
            mask_zero=True,
        )(encoder_inputs)

        encoder_outputs = keras_nlp.layers.TransformerEncoder(
            intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
        )(inputs=x)
        encoder = tf.keras.Model(encoder_inputs, encoder_outputs)

        # Decoder
        decoder_inputs = tf.keras.Input(
            shape=(None,), dtype="int64", name="decoder_inputs")
        encoded_seq_inputs = tf.keras.Input(
            shape=(None, EMBED_DIM), name="decoder_state_inputs")

        x = keras_nlp.layers.TokenAndPositionEmbedding(
            vocabulary_size=THA_VOCAB_SIZE,
            sequence_length=MAX_SEQUENCE_LENGTH,
            embedding_dim=EMBED_DIM,
            mask_zero=True,
        )(decoder_inputs)

        x = keras_nlp.layers.TransformerDecoder(
            intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
        )(decoder_sequence=x, encoder_sequence=encoded_seq_inputs)
        x = tf.keras.layers.Dropout(0.5)(x)
        decoder_outputs = tf.keras.layers.Dense(
            THA_VOCAB_SIZE, activation="softmax")(x)
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
    dataset = Dataset()
    model_builder = ModelBuilder()
    model = model_builder.build_model()

    model.summary()
    model.compile(
        "rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(dataset.train, epochs=10, validation_data=dataset.val)
