from typing import List, Dict

import tensorflow as tf
import keras_tuner as kt

from src.zephyrus.data_pipelines.parsers.Parser import Parser
from src.zephyrus.data_pipelines.utils import sin_cos_scale, vectorize, DAY_SECONDS, YEAR_SECONDS


class PointData(Parser):
    """
    Decoder for the point data

    Data layout
    A row(example) consists of a column per time step for each feature, delimited by the `$` symbol.
    There is a row for each plant in the dataset

    """

    static_features_s = ["ts", "plant", "DOY", "Hour", "Longitude", "Latitude"]
    lagged_features = ["Irradiance", "pres", "precip", "clouds", "wind_spd", "wind_dir", "temp", "rh", "ghi"]

    feats = {
        "plant": tf.string,
        "Irradiance": tf.float32,
        "Longitude": tf.float32,
        "Latitude": tf.float32,
        "pres": None,
        "precip": None,
        "clouds": None,
        "temp": None,
        "rh": None,
        "ghi": None
    }

    types = {
        "plant": tf.string,
        "Irradiance": tf.float32,
        "Longitude": tf.float32,
        "Latitude": tf.float32,
        "ts": tf.int64,
        "DOY": tf.int64,
        "Hour": tf.int64,
        "wind_dir": tf.int64,
        "wind_spd": tf.float32,
        "clouds": tf.int64,
        "pres": tf.float32,
        "precip": tf.float32,
        "temp": tf.float32,
        "rh": tf.float32,
        "ghi": tf.float32
    }

    def __init__(self, hp: kt.HyperParameters):
        self.hp = hp
        self.eval = eval

        static_features = {f"{c}": tf.io.FixedLenFeature([1], self.types[c]) for c in self.static_features_s}
        features_lagged = {f"{c}${l}": tf.io.FixedLenFeature([1], self.types[c]) for l in range(-24, 96) for c in
                           self.lagged_features}
        example_features = {**static_features, **features_lagged}
        self.features = example_features

    @property
    def runbook(self) -> List[Dict]:
        return [{"fn": self.vectorize_example, "op": "map"},
                {"fn": self.feature_transform, "op": "map"}]

    def vectorize_example(self, example):
        x = {}
        for c in self.lagged_features:
            stack = [example[f"{c}${l}".encode("ascii")] for l in range(-24, 96)]
            stack = tf.stack(stack)
            stack = tf.squeeze(stack)
            x[c] = stack

        for c in self.static_features_s:
            step = 60 * 60  # 1h in seconds
            time_steps = tf.range(-24, 96, dtype="int64")
            time_steps = time_steps * step

            x[c] = tf.squeeze([example[c.encode("ascii")]] * len(range(-24, 96)))
            if c == "ts":
                x[c] = x[c] + time_steps

        return x

    @staticmethod
    def feature_transform(x):
        # xt = {
        #     # Passthrough
        #     "plant": x["plant"],
        #     "ts": x["ts"],
        #     "irradiance": x["Irradiance"],
        #     "Longitude": x["Longitude"],
        #     "Latitude": x["Latitude"],
        #
        #     "clouds": x["clouds"],
        #     "pres": x["pres"],
        #     "precip": x["precip"],
        #     "temp": x["temp"],
        #     "rh": x["rh"],
        #     "ghi": x["ghi"]
        # }

        # Scale DOY and HOUR
        x["hour_sin"], x["hour_cos"] = sin_cos_scale(x["ts"], DAY_SECONDS)
        x["year_sin"], x["year_cos"] = sin_cos_scale(x["ts"], YEAR_SECONDS)

        # Vectorize wind
        x["wind_x"], x["wind_y"] = vectorize(x["wind_spd"], dir_deg=x["wind_dir"])

        y = x["Irradiance"]

        return x, y
