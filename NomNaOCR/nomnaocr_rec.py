"""NomNaOCR CRNN×CTC recognizer — reads a VERTICAL chữ-Nôm column → char sequence.

Rebuilds the exact ds4v/NomNaOCR CRNN architecture (custom_cnn + BiGRU×2 + CTC),
loads NomNaOCR_CRNNxCTC.h5, and CTC-decodes. Input is 432×48 with the sequence
along HEIGHT, so it reads a whole column TOP→BOTTOM — maps 1-1 to a kim column,
NO rotation needed.

Weights  : NomNaOCR/weights/NomNaOCR_CRNNxCTC.h5  (VOCAB_SIZE 7481 = 7479 chars +[PAD]+[UNK])
Vocab    : NomNaOCR/vocab.txt  (one char per line, most-common order) — build with build_vocab.py
Env      : the scratch tf_env (Python 3.10 + tensorflow 2.15). paddlepaddle/TF have no Py3.14 wheels.

Smoke (no vocab needed — just proves the weights load into our rebuilt arch):
    tf_env/bin/python nomnaocr_rec.py weights/NomNaOCR_CRNNxCTC.h5
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

HERE = Path(__file__).resolve().parent
# import the REPO's own layers so layer names match the .h5 exactly (load_weights by name)
sys.path.insert(0, str(HERE / "ds4v_repo" / "Text recognition"))
from layers import custom_cnn, reshape_features   # noqa: E402

HEIGHT, WIDTH = 432, 48
PAD = "[PAD]"
CONV_CFG = {
    "block1": {"num_conv": 1, "filters": 64,  "pool_size": (2, 2)},
    "block2": {"num_conv": 1, "filters": 128, "pool_size": (2, 2)},
    "block3": {"num_conv": 2, "filters": 256, "pool_size": (2, 2)},
    "block4": {"num_conv": 2, "filters": 512, "pool_size": (2, 2)},
    "block5": {"num_conv": 2, "filters": 512, "pool_size": None},
}


def build_crnn(vocab_size: int) -> tf.keras.Model:
    """Exact replica of CRNNxCTC.ipynb build_crnn (same layer names => weights map)."""
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Bidirectional, GRU
    image_input = Input(shape=(HEIGHT, WIDTH, 3), dtype="float32", name="image")
    x = custom_cnn(CONV_CFG, image_input)
    feature_maps = reshape_features(x, dim_to_keep=1, name="rnn_input")
    g1 = Bidirectional(GRU(256, return_sequences=True), name="bigru1")(feature_maps)
    g2 = Bidirectional(GRU(256, return_sequences=True), name="bigru2")(g1)
    y = Dense(vocab_size + 1, activation="softmax", name="rnn_output")(g2)  # +1 CTC blank
    return Model(inputs=image_input, outputs=y, name="CRNN")


def _resize_pad(pil_img):
    """distortion-free resize to (HEIGHT,WIDTH) preserving aspect, pad bottom/right
    with white, align top — matches loader.DataHandler.distortion_free_resize."""
    arr = tf.convert_to_tensor(np.asarray(pil_img.convert("RGB")), tf.float32)
    arr = tf.image.resize(arr, (HEIGHT, WIDTH), preserve_aspect_ratio=True)
    ph = HEIGHT - tf.shape(arr)[0]
    pw = WIDTH - tf.shape(arr)[1]
    arr = tf.pad(arr, [[0, ph], [0, pw], [0, 0]], constant_values=255.0)  # align top/left
    return arr / 255.0


class NomNaRecognizer:
    def __init__(self, weights_path, vocab_path, max_length: int = 30):
        vocab = [ln for ln in Path(vocab_path).read_text(encoding="utf-8").split("\n") if ln != ""]
        self.char2num = tf.keras.layers.StringLookup(vocabulary=vocab, mask_token=PAD)
        self.num2char = tf.keras.layers.StringLookup(
            vocabulary=self.char2num.get_vocabulary(), mask_token=PAD, invert=True)
        self.vocab_size = int(self.char2num.vocab_size())     # should be 7481
        self.max_length = max_length
        self.model = build_crnn(self.vocab_size)
        self.model.load_weights(str(weights_path))

    def recognize(self, pil_columns) -> list[str]:
        """List of PIL column images → list of decoded Nôm strings (top→bottom)."""
        if not pil_columns:
            return []
        batch = tf.stack([_resize_pad(im) for im in pil_columns])
        preds = self.model.predict(batch, verbose=0)
        input_len = tf.ones(len(preds)) * preds.shape[1]
        dec = tf.keras.backend.ctc_decode(preds, input_length=input_len, greedy=True)[0][0]
        dec = dec[:, : self.max_length].numpy()
        out = []
        for row in dec:
            toks = tf.constant([int(t) for t in row if int(t) > 1], dtype=tf.int64)  # drop PAD(0)/UNK(1)/blank(-1)
            chars = self.num2char(toks).numpy() if toks.shape[0] else []
            out.append("".join(c.decode("utf-8") for c in chars))
        return out


if __name__ == "__main__":
    # Smoke: build with the model's true VOCAB_SIZE (7481) and load weights — proves
    # our architecture replica matches the checkpoint, even without vocab.txt.
    w = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "weights" / "NomNaOCR_CRNNxCTC.h5")
    m = build_crnn(7481)                      # Dense = 7481+1 = 7482, matches the checkpoint
    m.load_weights(w)
    print(f"OK loaded weights into rebuilt CRNN (Dense units={m.get_layer('rnn_output').units}).")
    p = m.predict(tf.zeros((1, HEIGHT, WIDTH, 3)), verbose=0)
    print(f"inference OK: pred shape={p.shape} (timesteps={p.shape[1]}, classes={p.shape[2]}).")
    print("Need vocab.txt (build_vocab.py) to map class indices -> characters.")
