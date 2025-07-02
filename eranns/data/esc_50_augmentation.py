import os
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa


def load_and_crop(path, label, config):
    def f(p):
        p = p.numpy().decode("utf-8")
        y, _ = librosa.load(p, sr=config["SR"], mono=True)
        L = config["SR"] * config["TC"]
        y = y[:L]
        if y.shape[0] < L:
            pad = np.zeros(L - y.shape[0], dtype=np.float32)
            y = np.concatenate((y, pad))
        return y.astype(np.float32)
    audio = tf.py_function(f, [path], tf.float32)
    audio.set_shape((config["SR"] * config["TC"],))
    return audio, label

def _load_crop(path, label, config):
    y, lbl = tf.py_function(
        lambda p, l: load_and_crop(p, l, config), [path, label], [tf.float32, tf.int32]
    )
    y.set_shape([config["SR"] * config["TC"]])
    lbl.set_shape([])
    return y, lbl

def audio_to_mel(y, label, config):
    def f(arr):
        arr = np.array(arr)
        S = librosa.feature.melspectrogram(
            y=arr, sr=config["SR"], n_fft=config["WINDOW"], hop_length=config["HOP"],
            n_mels=config["NUM_MELS"], fmin=config["FMIN"], fmax=config["FMAX"], power=2.0
        )
        S += 1e-10
        db = librosa.power_to_db(S, ref=np.max).astype(np.float32)
        db = (db + 80) / 80
        return db[..., np.newaxis]
    mel = tf.py_function(f, [y], tf.float32)
    mel.set_shape((config["NUM_MELS"], config["Ts"], 1))
    return mel, label

def spec_augment(mel, label, config):
    for _ in range(2):
        t0 = tf.random.uniform([], 0, config["Ts"] - 8 * config["TC"], tf.int32)
        w = tf.random.uniform([], 0, 8 * config["TC"], tf.int32)
        mel = tf.concat([
            mel[:, :t0, :],
            tf.zeros([config["NUM_MELS"], w, 1], dtype=mel.dtype),
            mel[:, t0+w:, :]
        ], axis=1)
    for _ in range(2):
        f0 = tf.random.uniform([], 0, config["NUM_MELS"] - 16, tf.int32)
        w = tf.random.uniform([], 0, 16, tf.int32)
        mel = tf.concat([
            mel[:f0, :, :],
            tf.zeros([w, config["Ts"], 1], dtype=mel.dtype),
            mel[f0+w:, :, :]
        ], axis=0)
    return mel, label

def pitch_and_crop(path, label, config):
    def f(p):
        p = p.numpy().decode("utf-8")
        y, _ = librosa.load(p, sr=config["SR"], mono=True)
        steps = np.random.randint(-2, 3)
        y2 = librosa.effects.pitch_shift(y, sr=config["SR"], n_steps=steps)
        L = config["SR"] * config["TC"]
        if len(y2) < L:
            y2 = np.pad(y2, (0, L-len(y2)), "constant")
        else:
            i = np.random.randint(0, len(y2)-L+1)
            y2 = y2[i:i+L]
        return y2.astype(np.float32)
    
    y = tf.py_function(f, [path], tf.float32)
    y.set_shape([config["SR"] * config["TC"]])
    return y, label

def mixup(ds, alpha=1.0, config=None):
    def _mix(a, b):
        x1, y1 = a; x2, y2 = b
        beta = tf.random.gamma([], alpha, 1.0)
        x = beta * x1 + (1-beta) * x2
        y = beta * tf.cast(y1, tf.float32) + (1-beta) * tf.cast(y2, tf.float32)
        return x, y
    ds1 = ds.shuffle(1024)
    ds2 = ds.shuffle(1024)
    return tf.data.Dataset.zip((ds1, ds2)).map(_mix, num_parallel_calls=config["AUTOTUNE"])

def build_augmented_dataset(paths, labels, config):
    base = tf.data.Dataset.from_tensor_slices((paths, labels)).cache()
    AUTOTUNE = config["AUTOTUNE"]

    ds_crop = (
        base.map(lambda x, y: _load_crop(x, y, config), AUTOTUNE)
            .map(lambda y, lbl: (y, tf.one_hot(tf.cast(lbl, tf.int32), depth=config["NUM_CLASSES"])), AUTOTUNE)
            .map(lambda y, lbl: audio_to_mel(y, lbl, config), AUTOTUNE)
    )

    ds_pitch = (
        base.map(lambda x, y: pitch_and_crop(x, y, config), AUTOTUNE)
            .map(lambda y, lbl: (y, tf.one_hot(tf.cast(lbl, tf.int32), depth=config["NUM_CLASSES"])), AUTOTUNE)
            .map(lambda y, lbl: audio_to_mel(y, lbl, config), AUTOTUNE)
    )

    ds_specaug = ds_crop.map(lambda mel, lbl: spec_augment(mel, lbl, config), AUTOTUNE)
    ds_mix = mixup(ds_crop, config=config)

    ds_all = (
        ds_crop.concatenate(ds_pitch)
               .concatenate(ds_specaug)
               .concatenate(ds_mix)
               .shuffle(2048)
               .batch(config["BATCH"])
               .prefetch(AUTOTUNE)
    )

    return ds_all

def create_esc50_augmentation(csv_path, audio_dir, fold_val, config):
    meta = pd.read_csv(csv_path)
    df_train = meta[meta.fold != fold_val]
    df_val = meta[meta.fold == fold_val]

    tr_p = df_train.filename.map(lambda f: os.path.join(audio_dir, f)).tolist()
    tr_l = df_train.target.astype(np.int32).tolist()
    va_p = df_val.filename.map(lambda f: os.path.join(audio_dir, f)).tolist()
    va_l = df_val.target.astype(np.int32).tolist()

    ds_tr = build_augmented_dataset(tr_p, tr_l, config)
    ds_va = (
        tf.data.Dataset.from_tensor_slices((va_p, va_l))
            .map(lambda x, y: _load_crop(x, y, config), config["AUTOTUNE"])
            .map(lambda y, lbl: (y, tf.one_hot(tf.cast(lbl, tf.int32), depth=config["NUM_CLASSES"])), config["AUTOTUNE"])
            .map(lambda y, lbl: audio_to_mel(y, lbl, config), config["AUTOTUNE"])
            .batch(config["BATCH"])
            .prefetch(config["AUTOTUNE"])
    )

    return ds_tr, ds_va
