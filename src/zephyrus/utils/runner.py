import argparse
import json
import sys
from abc import ABC, abstractmethod
from multiprocessing.pool import ThreadPool
from typing import List, Dict, Tuple

import pandas as pd
import tensorflow as tf
import functools
import os
import glob
from urllib.parse import unquote
from aletheia import Metastore
import zephyrus.utils.translator as t
from zephyrus.utils.hyperparameters import HyperParameters_Extend
from zephyrus.utils.standard_logger import build_logger


def arg_parser(parser=None) -> argparse.ArgumentParser:
    parser = parser if parser else argparse.ArgumentParser(description="Runner of Models")
    parser.add_argument('--name', type=str, help="Runner Name, included in logs")
    parser.add_argument('--output_dir', default="./res", type=str, help="Folder to save results")
    parser.add_argument('--min_date', default=False, action='store_true', help="filter data on 2018")
    parser.add_argument('--per_plant', default=False, action='store_true', help="Run per plant or all at once")
    parser.add_argument('--threads', default=16, type=int, help="Number of threads per model")
    parser.add_argument('--par_models', default=1, type=int, help="Number models to run in parallel")
    return parser


class Runner(ABC):
    sp = os.path.join(t.get_path("data"), "point_irrad_weather(17-19).dataset", f"part=*")

    static_features_s = ["ts", "plant", "DOY", "Hour", "Longitude", "Latitude"]
    # "hour_sin", "hour_cos", "year_sin", "year_cos"

    feats = ["irradiance", "pres", "precip", "clouds", "temp", "rh",
             "ghi", "wind_x", "wind_y", "hour_sin", "hour_cos", "year_sin", "year_cos"]

    def __init__(self, output_dir="", hp: HyperParameters_Extend = None, per_plant=True, min_date=False,
                 threads: int = 16, par_models: int = 1, name=None, drop_remainder=False, **kwargs):
        name = __name__ if name is None else name
        self.logger = build_logger(name)

        # Settings
        self.output_dir = output_dir
        self.par_models = par_models
        self.threads = threads
        self.per_plant = per_plant
        self.min_date = min_date
        self.drop_remainder = drop_remainder
        self.hp = hp
        # self.model_settings = {}

        self._ms = Metastore("zephyrus", gcp_project="alethea-fcf82")  # Get a projects metastore

        # Get all the plants
        plants = self._ms.get_metadata("plant_folds")
        # Remove bad plants
        plants.pop('clapham')
        plants.pop('soho farm')
        self.plants = plants

        self.logger.info("Setting up plant datasets")

        self._ds_path = glob.glob(self.sp)
        self.logger.info(f"Reading data from {self.sp}")
        self.min_date_filter_ts = 1535756400
        self.test_train_split_ts = 1556668800

        self._plant_datasets = self.make_plant_data_set(self._ds_path)
        self._all_datasets = self.make_test_train_data_set(self._ds_path)

    @abstractmethod
    def feature_extract(self, x, y):
        pass


    def make_test_train_data_set(self, ds_path, filters: List = None) -> (tf.data.Dataset, tf.data.Dataset):
        if filters is None:
            filters = []

        if self.min_date:
            filters.append(lambda x, *_: x["ts"][0] > self.min_date_filter_ts)

        test_train: tf.data.Dataset = functools.reduce(tf.data.Dataset.concatenate,
                                                       [tf.data.experimental.load(fp, compression="GZIP") for fp in
                                                        ds_path])

        # Split on
        train_f = (lambda x, *_: x["ts"][0] < self.test_train_split_ts)
        test_f = (lambda x, *_: x["ts"][0] > self.test_train_split_ts)

        test_train = functools.reduce(tf.data.Dataset.filter, filters, test_train)

        train = test_train.filter(train_f)\
            .map(self.feature_extract, num_parallel_calls=tf.data.AUTOTUNE)\
            .batch(512, drop_remainder=self.drop_remainder)

        test = test_train.filter(test_f)\
            .map(self.feature_extract, num_parallel_calls=tf.data.AUTOTUNE)\
            .batch(512, drop_remainder=self.drop_remainder)
        return (train, test)

    def make_plant_data_set(self, ds_path) -> Dict[str, Tuple[tf.data.Dataset, tf.data.Dataset]]:
        datasets = {}
        for p in self.plants:
            plant_filter = (lambda x, *_: x["plant"][0] == unquote(p).encode("ascii"))
            datasets[p] = self.make_test_train_data_set(ds_path, filters=[plant_filter])
        return datasets

    @abstractmethod
    def make_model(self) -> tf.keras.Model:
        pass

    @abstractmethod
    def fit_model(self, m: tf.keras.Model, d: tf.data.Dataset):
        return m.fit(d)

    @abstractmethod
    def eval_model(self, model, test) -> pd.DataFrame:
        """
        Use the given test data to evaluate the model and return its predictions in a pd.Dataframe
        Args:
            model:
            test:

        Returns:

        """
        pass

    def settings_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k[0] != "_" and k != "logger"}

    def save_settings(self) -> None:
        self.logger.info(f"Save Settings -- {self.output_dir}")
        os.makedirs(os.path.join(t.get_path("results"), self.output_dir), exist_ok=True)
        out_file = os.path.join(t.get_path("results"), self.output_dir, "settings.json")
        with open(out_file, "w") as f:
            settings_dict = self.settings_dict()
            settings = json.dumps(settings_dict, skipkeys=True, indent=4, sort_keys=True, default=lambda o:o.__dict__())
            f.write(settings)

    def save_df(self, df: pd.DataFrame, output_folder: str) -> None:
        self.logger.info(f"Save -- {output_folder}")
        os.makedirs(os.path.join(t.get_path("results"), output_folder), exist_ok=True)
        out_file = os.path.join(t.get_path("results"), output_folder, "res.pqt")
        self.logger.info(f"Writing -- {out_file}")
        df.to_parquet(out_file)

    def _run_plant(self, plant: str):
        self.logger.info(f"Running for {plant}")
        ds = self._plant_datasets[plant]
        out_dir = os.path.join(self.output_dir, f"plant={plant}")
        self._run_all(ds, out_dir)

    def _run_on_plants(self):
        self.logger.info(f"Starting Thread Pool: {self.par_models} Thread(s)")
        with ThreadPool(self.par_models) as p:
            res = p.map(self._run_plant, self.plants)
            r = list(res)
            print(r)

    def _run_all(self, dataset, output_folder):
        train, test = dataset
        self.logger.info(f"Fitting -- {output_folder}")
        model = self.make_model()
        self.fit_model(model, train)

        self.logger.info(f"Eval and Save -- {output_folder}")
        res = self.eval_model(model, test)
        self.save_df(res, output_folder)

    def run(self):
        self.save_settings()
        if self.per_plant:
            self.logger.info(f"Fitting per plants")
            self._run_on_plants()
        else:
            self.logger.info(f"Fitting on all")
            self._run_all(self._all_datasets, self.output_dir)

    @staticmethod
    def arg_parser(ap):
        pass
