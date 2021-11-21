from abc import ABC,abstractmethod
from typing import Dict, Callable, List
import tensorflow as tf

class Parser(ABC):
    @property
    @abstractmethod
    def runbook(self) -> List[Dict]:
        pass

    def run(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        for r in self.runbook:
            op, func = r["op"], r["fn"]
            ds = self.PIPELINE_STEPS[op](ds, func)
        return ds

    @staticmethod
    def _ds_map(ds: tf.data.Dataset, f):
        return ds.map(f, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    @staticmethod
    def _ds_filter(ds: tf.data.Dataset, f):
        return ds.filter(f)

    @staticmethod
    def _ds_inter(ds: tf.data.Dataset, f):
        return ds.interleave(f, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    PIPELINE_STEPS = {
        "map": _ds_map,
        "filter": _ds_filter,
        "interleave": _ds_inter
    }