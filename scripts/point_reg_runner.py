import glob
import os
import sys
import tensorflow as tf
import pandas as pd
# from aletheia import Metastore
import keras_tuner as kt
from tensorflow.keras import mixed_precision

from zephyrus.models.loss import ZeroLoss
from zephyrus.point_reg.auto_reg import build_auto_reg
from zephyrus.utils.standard_logger import logger
from zephyrus.utils import translator as t, hyperparameters
from zephyrus.data_pipelines import pipeline
from zephyrus.data_pipelines.parsers.PointDataPQT import PointData


logger.info(f"Tensorflow Version: {tf.__version__}")
logger.info(f"Setting compute mode to 'mixed_float16'")
mixed_precision.set_global_policy('mixed_float16')

hp = hyperparameters.HyperParameters_Extend()
hp.Fixed("BATCH_SIZE", 512)
 # ~ 512 = 670ms

# Range
hp.Fixed("USE_WEATHER", True)
hp.Fixed("WARMUP_STEPS", 24) # used as a slices so will take e.g [0, 1, ..., 23]
hp.Fixed("OUTPUT_STEPS", 24)
hp.Fixed("OUTPUT_STEPS_start", hp.get("WARMUP_STEPS") - 25 + 12) # these are offset by 24 in the data read pipeline
hp.Fixed("OUTPUT_STEPS_end", hp.get("WARMUP_STEPS") + hp.get("OUTPUT_STEPS") - 24 + 12) # these are offset by 24 in the data read pipeline

raw_data = os.path.join(t.get_path("data"), "irrad_tree.tfrecord/part*")
raw_data_paths = glob.glob(raw_data)

parser_test = PointData(hp)


def reshape(x,y):
    return ({**x, "out_ts": x["ts"][:, hp.get("OUTPUT_STEPS_start") + 24: hp.get("OUTPUT_STEPS_end") + 24]},
     y[:, hp.get("OUTPUT_STEPS_start") + 24: hp.get("OUTPUT_STEPS_end") + 24])


data_file = "point_irrad_weather(17-19)"
train = pipeline.data_load_batch(data_file, hp, data_folder=t.get_path("data"), cache=True).map(reshape, num_parallel_calls=8)
test = pipeline.data_load_batch(data_file, hp, data_folder=t.get_path("data"), cache=True).map(reshape, num_parallel_calls=8)

expr = f"20211118-moredata"
logger.info(f"Exper: {expr}")

exper_path = os.path.join(t.get_path("results"), expr)
ckpt_path = os.path.join(exper_path, "lstm-ckpt/ckpt-{epoch:02d}")
logger.info(f"Ckpt path: {ckpt_path}")
ckpt = tf.keras.callbacks.ModelCheckpoint(
      filepath=ckpt_path,
      save_weights_only=True,
      save_best_only=False)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tb_path = os.path.join(exper_path, "tb")
tb = tf.keras.callbacks.TensorBoard(log_dir=tb_path, profile_batch=(20,25))
os.makedirs(tb_path, exist_ok=True)

ckpt_dir, _ = os.path.split(ckpt_path)
latest = tf.train.latest_checkpoint(ckpt_dir)


def fit(m, test, train, callbacks):
    m.compile(loss="MSE", metrics=[ZeroLoss(step=0, name="t0"), ZeroLoss(step=1, name="t1"), ZeroLoss(step=-1, name="te")])
    m.fit(train, validation_data=test, callbacks=callbacks, epochs=50)
    return m


m = build_auto_reg(hp)
callbacks = [ckpt,tb]

if latest:
    logger.info(f"ckpt found, restoring {latest}")
    m.load_weights(latest)
# else:
m = fit(m, test, train, callbacks)

def save_pred(m, data, name):
    def same_size(x, y, yhat):
        out = hp.get("OUTPUT_STEPS") + 1
        return {"plant": x["plant"][:out], "ts": x["out_ts"], "step": tf.range(out), "y": y, "yhat": yhat}

    results = data.map(lambda x,y: (x, y, m(x)), num_parallel_calls=tf.data.AUTOTUNE).unbatch()
    results_flat = results.map(same_size, num_parallel_calls=tf.data.AUTOTUNE)
    results_flat = results_flat.unbatch()


    results_pd_df = pd.DataFrame(results_flat.as_numpy_iterator())
    save_path = os.path.join(t.get_path("results"), expr, name)
    results_pd_df.to_parquet(save_path)
    logger.info("wrote output")

save_pred(m, train, "train_res.pqt")
save_pred(m, train, "test_res.pqt")
