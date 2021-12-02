import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.ensemble import RandomForestRegressor
from zephyrus.utils.runner import Runner, arg_parser as _ap


class SkitTreeRunner(Runner):
    def __init__(self, warmup_steps=24, y_offset=0, **kwargs):
        self.warmup_steps = warmup_steps
        self.y_offset = y_offset
        super().__init__(**kwargs)

    def feature_extract(self, x, y):
        return {f: x[f][:self.warmup_steps] for f in self.feats}, y[self.y_offset + self.warmup_steps]

    def dataset_to_df(self, ds):
        ds = ds.unbatch()
        dx = ds.map(lambda x, y: x)
        dy = ds.map(lambda x, y: {"y": y})

        df_x = tfds.as_dataframe(dx)
        df_x = pd.DataFrame(df_x)
        cols = df_x.columns
        for c in cols:
            df_x[[f"{c}${i}" for i in range(-24, 0)]] = pd.DataFrame(df_x[c].tolist())
        df_x = df_x.drop(cols, axis=1)

        df_y = tfds.as_dataframe(dy)
        df_y = pd.DataFrame(df_y)

        return df_x, df_y

    def make_model(self):
        model_settings = {"n_jobs": -1, "max_depth": None, "min_samples_leaf": 2, "n_estimators": 500,
                          "max_features": "auto"}
        model = RandomForestRegressor(**model_settings)
        return model

    def fit_model(self, m, d: tf.data.Dataset):
        df = self.dataset_to_df(d)
        x, y = df
        return m.fit(x, y)

    def eval_model(self, model, test):
        x, y = self.dataset_to_df(test)
        y["y_hat"] = model.predict(x)
        return y

    @staticmethod
    def arg_parser(ap=None):
        ap = ap if ap else _ap()
        ap.add_argument('--warmup_steps', default=24, type=int, help="Number of steps to see before making prediction")
        ap.add_argument('--y_offset', default=0, type=int, help="Number of steps after warmup_steps to "
                                                                    "leave before Y (prediction) .e.g if warmup steps=24 "
                                                                    "and y_offset is 0, the model will use (0-23] to "
                                                                    "predict step 24")
        return ap

if __name__ == "__main__":
    # Parse Args
    parser = SkitTreeRunner.arg_parser()
    args = parser.parse_args()
    print(args)
    tr = SkitTreeRunner(**args.__dict__)
    tr.run()