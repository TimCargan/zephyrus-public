import os
import tensorflow as tf
from zephyrus.data_pipelines.parsers.Parser import Parser
from zephyrus.utils.standard_logger import logger


def load(paths, parser:Parser, compression_type="GZIP", shuffle=True, shuffle_buff=100, batch=True, batch_size=32, drop_remainder=False):
    at = tf.data.AUTOTUNE
    files = tf.data.Dataset.from_tensor_slices(paths)
    ds = tf.data.TFRecordDataset(files, buffer_size=int(500e6),
                                      compression_type=compression_type,
                                      num_parallel_reads=at)
    ds = parser.run(ds)
    ds = ds.prefetch(at)

    ds = ds.shuffle(buffer_size=shuffle_buff) if shuffle else ds
    ds = ds.batch(batch_size=batch_size, drop_remainder=drop_remainder) if batch else ds
    return ds


def save(paths, parser, name, data_folder="/mnt/d/data/", compression="GZIP"):
    dataset = load(paths, parser, shuffle=False, batch=False)
    save_path = os.path.join(data_folder, f"{name}.dataset")
    logger.info("saving start")
    tf.data.experimental.save(dataset, save_path, compression=compression)
    logger.info("saving done")
    return dataset


def data_load(name, data_folder="/mnt/d/data/", filter=None, cache=False):
    dataset = tf.data.experimental.load(f"{data_folder}/{name}.dataset", compression="GZIP")
    dataset = dataset.filter(filter) if filter else dataset
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache() if cache else dataset
    return dataset


def data_load_batch(name, hp, drop_remainder=False, data_folder="/mnt/d/data/", filter=None, cache=False):
    dataset = data_load(name, data_folder=data_folder, filter=filter, cache=cache)
    dataset = dataset.shuffle(buffer_size=100 * hp.get("BATCH_SIZE")) \
      .batch(hp.get("BATCH_SIZE"), drop_remainder=drop_remainder,  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset
