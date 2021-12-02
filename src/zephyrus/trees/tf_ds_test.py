import argparse
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf
from zephyrus.utils.runner import Runner, arg_parser as _ap


class TfdfRunner(Runner):
    def __init__(self, warmup_steps=24, y_offset=0, **kwargs):
        self.warmup_steps = warmup_steps
        self.y_offset = y_offset
        self.tree_settings = {"num_trees": 500, "min_examples": 2, "max_depth": 32, "verbose": False}
        super().__init__(**kwargs)

    def feature_extract(self, x, y):
        return {f: x[f][:self.warmup_steps] for f in self.feats}, y[self.y_offset + self.warmup_steps]

    def make_model(self) -> tf.keras.Model:
        model = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION, num_threads=self.threads,
                                             **self.tree_settings)
        model.compile()
        return model

    def fit_model(self, m: tf.keras.Model, d: tf.data.Dataset):
        return m.fit(d)

    def eval_model(self, model, test):
        dfs = []
        for x, y in test:
            y_hat = tf.squeeze(model.predict(x))
            t_df = pd.DataFrame({"y": y, "y_hat": y_hat, "step": self.y_offset})
            dfs.append(t_df)

        df = pd.concat(dfs, axis=0)
        return df

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
    parser = TfdfRunner.arg_parser()
    args = parser.parse_args()
    print(args)
    tr = TfdfRunner(name=args.name,
                    threads=args.threads, output_dir=args.output_dir,
                    per_plant=args.per_plant, min_date=args.min_date)
    tr.run()
