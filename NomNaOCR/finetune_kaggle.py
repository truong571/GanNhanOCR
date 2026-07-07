"""Fine-tune NomNaOCR CRNN×CTC on the Sách data — self-contained, runs on Kaggle/Colab GPU.

Transfer learning: rebuild the head to the Sách vocab (covers all 1591 chars; the
pretrained 7481-head only covers 82%), load the CNN+BiGRU from NomNaOCR_CRNNxCTC.h5 by
name (skip the mismatched head), then train with CTC. Saves the fine-tuned weights +
the NEW vocab (so nomnaocr_rec.py can decode).

Reuses the repo's proven pieces (CTCLoss, DataImporter/DataHandler, layers) from
'Text recognition/'. Point --repo at that folder (upload it to Kaggle too).

KAGGLE (add both as datasets):
  finetune_data  -> /kaggle/input/sach-finetune/Datasets/Patches
  weights + repo -> /kaggle/input/nomna-assets/{NomNaOCR_CRNNxCTC.h5, Text recognition/}
Run in a GPU notebook:
  !python finetune_kaggle.py \
     --dataset_dir /kaggle/input/sach-finetune/Datasets/Patches \
     --pretrained  /kaggle/input/nomna-assets/NomNaOCR_CRNNxCTC.h5 \
     --repo "/kaggle/input/nomna-assets/Text recognition" \
     --epochs 30 --batch 64 --out /kaggle/working
Outputs: /kaggle/working/finetuned_CRNNxCTC.h5  +  finetuned_vocab.txt  (download both).

LOCAL SMOKE (tf_env, CPU): add --limit 200 --epochs 1  -> just proves it runs.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import tensorflow as tf

HEIGHT, WIDTH = 432, 48
PAD = "[PAD]"
CONV_CFG = {
    "block1": {"num_conv": 1, "filters": 64,  "pool_size": (2, 2)},
    "block2": {"num_conv": 1, "filters": 128, "pool_size": (2, 2)},
    "block3": {"num_conv": 2, "filters": 256, "pool_size": (2, 2)},
    "block4": {"num_conv": 2, "filters": 512, "pool_size": (2, 2)},
    "block5": {"num_conv": 2, "filters": 512, "pool_size": None},
}


def build_crnn(vocab_size, custom_cnn, reshape_features):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Bidirectional, GRU
    inp = Input(shape=(HEIGHT, WIDTH, 3), dtype="float32", name="image")
    x = custom_cnn(CONV_CFG, inp)
    feat = reshape_features(x, dim_to_keep=1, name="rnn_input")
    g1 = Bidirectional(GRU(256, return_sequences=True), name="bigru1")(feat)
    g2 = Bidirectional(GRU(256, return_sequences=True), name="bigru2")(g1)
    y = Dense(vocab_size + 1, activation="softmax", name="rnn_output")(g2)
    return Model(inp, y, name="CRNN")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True, help="folder with All.txt/Validate.txt + <book>/*.jpg")
    ap.add_argument("--pretrained", required=True, help="NomNaOCR_CRNNxCTC.h5 (init)")
    ap.add_argument("--repo", required=True, help="path to the repo's 'Text recognition' dir")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1.0)         # Adadelta
    ap.add_argument("--out", default=".")
    ap.add_argument("--limit", type=int, default=0, help="cap samples (local smoke only)")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    from loader import DataImporter, DataHandler          # noqa: E402
    from losses import CTCLoss                            # noqa: E402
    from layers import custom_cnn, reshape_features       # noqa: E402
    dsd = Path(args.dataset_dir)

    train = DataImporter(str(dsd), str(dsd / "All.txt"), min_length=1)
    val = DataImporter(str(dsd), str(dsd / "Validate.txt"), min_length=1)
    print(train)
    dh = DataHandler(train, img_size=(HEIGHT, WIDTH), padding_char=PAD)
    dh.max_length = max(dh.max_length, max((len(l) for l in val.labels), default=0))

    def make_ds(paths, labels, shuffle):
        n = len(paths) if not args.limit else min(args.limit, len(paths))
        ds = tf.data.Dataset.from_tensor_slices((paths[:n], labels[:n]))
        if shuffle:
            ds = ds.shuffle(min(n, 4096))
        ds = ds.map(lambda p, l: (dh.process_image(p), dh.process_label(l)),
                    num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(args.batch).prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(train.img_paths, train.labels, True)
    val_ds = make_ds(val.img_paths, val.labels, False)

    VS = int(dh.char2num.vocab_size())
    print(f"vocab_size (with PAD/UNK) = {VS}  -> Dense = {VS + 1}")
    model = build_crnn(VS, custom_cnn, reshape_features)
    model.load_weights(args.pretrained, by_name=True, skip_mismatch=True)  # transfer CNN+BiGRU, reinit head
    model.compile(optimizer=tf.keras.optimizers.Adadelta(args.lr), loss=CTCLoss())

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(out / "finetuned_CRNNxCTC.h5"),
                                           monitor="val_loss", save_best_only=True, save_weights_only=True),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs, verbose=1)
    model.save_weights(str(out / "finetuned_CRNNxCTC.h5"))
    # save the NEW vocab (most-common order, no PAD/UNK) for nomnaocr_rec.py
    (out / "finetuned_vocab.txt").write_text("\n".join(train.vocabs), encoding="utf-8")
    print(f"\nSAVED -> {out}/finetuned_CRNNxCTC.h5  +  finetuned_vocab.txt  ({len(train.vocabs)} chars)")


if __name__ == "__main__":
    main()
