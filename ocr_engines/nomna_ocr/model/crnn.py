"""Ported from NomNaSite (ds4v/NomNaSite) — Streamlit decorators removed,
vocab path made configurable so the class can be instantiated outside
the original app layout.
"""

from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import (
    Bidirectional, Dense, GRU, Input, MaxPool2D, Reshape,
)

from .layers import ConvBnRelu


class CRNN(tf.keras.Model):
    def __init__(self, vocab_path: str | Path):
        super().__init__()
        self.max_length = 24
        self.height, self.width = 432, 48

        vocab_path = Path(vocab_path)
        with open(vocab_path, encoding="utf-8") as f:
            vocab = f.read().splitlines()

        self.num2char = tf.keras.layers.StringLookup(
            vocabulary=vocab,
            mask_token="[PAD]",
            invert=True,
        )
        self.model = self._build_model()

    def _build_model(self):
        image_input = Input(
            shape=(self.height, self.width, 3),
            dtype="float32",
            name="image",
        )
        x = ConvBnRelu(64, 3, name="block1_convbn")(image_input)
        x = MaxPool2D((2, 2), name="block1_pool")(x)

        x = ConvBnRelu(128, 3, name="block2_convbn")(x)
        x = MaxPool2D((2, 2), name="block2_pool")(x)

        x = ConvBnRelu(256, 3, name="block3_convbn1")(x)
        x = ConvBnRelu(256, 3, name="block3_convbn2")(x)
        x = MaxPool2D((2, 2), name="block3_pool")(x)

        x = ConvBnRelu(512, 3, name="block4_convbn1")(x)
        x = ConvBnRelu(512, 3, name="block4_convbn2")(x)
        x = MaxPool2D((2, 2), name="block4_pool")(x)

        x = ConvBnRelu(512, 2, padding="valid", name="block5_convbn1")(x)
        x = ConvBnRelu(512, 2, padding="valid", name="block5_convbn2")(x)

        # Keras 3 KerasTensors expose `.shape` (tuple) but not `.get_shape()`.
        _, height, width, channel = x.shape
        feature_maps = Reshape(
            target_shape=((height, width * channel)), name="rnn_input",
        )(x)

        bigru1 = Bidirectional(GRU(256, return_sequences=True), name="bigru1")(
            feature_maps,
        )
        bigru2 = Bidirectional(GRU(256, return_sequences=True), name="bigru2")(bigru1)

        y_pred = Dense(
            self.num2char.vocabulary_size() + 1,
            activation="softmax",
            name="rnn_output",
        )(bigru2)
        return tf.keras.Model(inputs=image_input, outputs=y_pred, name="CRNN")

    def distortion_free_resize(self, image, align_top=True):
        image = tf.image.resize(image, size=(self.height, self.width),
                                preserve_aspect_ratio=True)
        pad_height = self.height - tf.shape(image)[0]
        pad_width = self.width - tf.shape(image)[1]
        if pad_height == 0 and pad_width == 0:
            return image

        if pad_height % 2 != 0:
            h = pad_height // 2
            pad_height_top, pad_height_bottom = h + 1, h
        else:
            pad_height_top = pad_height_bottom = pad_height // 2

        if pad_width % 2 != 0:
            w = pad_width // 2
            pad_width_left, pad_width_right = w + 1, w
        else:
            pad_width_left = pad_width_right = pad_width // 2

        return tf.pad(
            image,
            paddings=[
                [0, pad_height_top + pad_height_bottom] if align_top
                else [pad_height_top, pad_height_bottom],
                [pad_width_left, pad_width_right],
                [0, 0],
            ],
            constant_values=255,
        )

    def process_image(self, image, img_align_top=True):
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = self.distortion_free_resize(image, img_align_top)
        image = tf.cast(image, tf.float32) / 255.0
        return image

    def ctc_decode(self, predictions, max_length):
        # Rewritten against `tf.nn.ctc_greedy_decoder` (stable TF op) so it
        # works with TF 2.16+/Keras 3 where `tf.keras.backend.ctc_decode`
        # is deprecated/removed. Semantics match the original NomNaSite
        # implementation: greedy CTC, then map [UNK] (token id 1) to -1
        # so it is filtered along with blanks in `tokens2texts`.
        batch_size = tf.shape(predictions)[0]
        timesteps = tf.shape(predictions)[1]
        input_length = tf.fill([batch_size], timesteps)

        predictions_t = tf.transpose(predictions, [1, 0, 2])
        decoded, _ = tf.nn.ctc_greedy_decoder(predictions_t, input_length)
        dense = tf.sparse.to_dense(decoded[0], default_value=-1)
        dense = dense[:, :max_length]

        return tf.where(
            dense == tf.cast(1, dense.dtype),
            tf.cast(-1, dense.dtype),
            dense,
        )

    def tokens2texts(self, batch_tokens):
        batch_texts = []
        batch_tokens = self.ctc_decode(batch_tokens, self.max_length)

        for tokens in batch_tokens:
            indices = tf.gather(
                tokens, tf.where(tf.logical_and(tokens != 0, tokens != -1))
            )
            text = tf.strings.reduce_join(self.num2char(indices))
            text = text.numpy().decode("utf-8")
            batch_texts.append(text)
        return batch_texts

    def predict_one_patch(self, patch_img) -> str:
        image = self.process_image(patch_img)
        pred_tokens = self.model.predict(
            tf.expand_dims(image, axis=0), verbose=0,
        )
        pred_labels = self.tokens2texts(pred_tokens)
        return pred_labels[0]
