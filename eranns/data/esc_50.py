import os
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa

def _load_log_mel(path, config):
    """
    Carrega um mel-spectrograma a partir do caminho do arquivo de áudio.
    """
    path = path.numpy().decode("utf-8")
    audio, _ = librosa.load(path, sr=config['SR'], mono=True)

    t = len(audio) / config['SR']
    t_int = int(np.ceil(t))
    target_frames = 128 * t_int
    required_len = config['HOP'] * (target_frames - 1) + config['WINDOW']

    if len(audio) < required_len:
        pad_amt = required_len - len(audio)
        audio = np.pad(audio, (0, pad_amt), mode="constant")
    else:
        audio = audio[:required_len]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=config['SR'],
        n_fft=config['WINDOW'],
        hop_length=config['HOP'],
        n_mels=config['NUM_MELS'],
        fmin=config['FMIN'],
        fmax=config['FMAX'],
        power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db.astype(np.float32)[..., np.newaxis]
    return mel_db

def _prep_mel(config):
    def wrapper(path, label):
        mel = tf.py_function(lambda p: _load_log_mel(p, config), [path], tf.float32)
        mel.set_shape((config['NUM_MELS'], None, 1))
        return mel, label
    return wrapper

def create_esc50(csv_path, audio_dir, fold_val, config):
    meta = pd.read_csv(csv_path)
    df_train = meta[meta.fold != fold_val]
    df_val   = meta[meta.fold == fold_val]

    train_paths  = df_train.filename.apply(lambda f: os.path.join(audio_dir, f)).tolist()
    train_labels = df_train.target.values.astype(np.int32)

    val_paths    = df_val.filename.apply(lambda f: os.path.join(audio_dir, f)).tolist()
    val_labels   = df_val.target.values.astype(np.int32)

    ds_train = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    ds_val   = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))

    one_hot = lambda y, lbl: (y, tf.one_hot(tf.cast(lbl, tf.int32), depth=config['NUM_CLASSES']))

    ds_train = (
        ds_train
        .map(one_hot, config['AUTOTUNE'])
        .map(_prep_mel(config), num_parallel_calls=config['AUTOTUNE'])
        .shuffle(config['SHUFFLE_SIZE'])
        .batch(config['BATCH_SIZE'])
        .prefetch(config['AUTOTUNE'])
    )

    ds_val = (
        ds_val
        .map(one_hot, config['AUTOTUNE'])
        .map(_prep_mel(config), num_parallel_calls=config['AUTOTUNE'])
        .batch(config['BATCH_SIZE'])
        .prefetch(config['AUTOTUNE'])
    )

    return ds_train, ds_val
