import tensorflow as tf

from zephyrus.trees.scikit_tree import SkitTreeRunner
from zephyrus.trees.tf_ds_test import TfdfRunner
from zephyrus.utils.runner import arg_parser, Runner
from zephyrus.utils.standard_logger import logger
from zephyrus.utils.hyperparameters import HyperParameters_Extend as HP

if __name__ == "__main__":
    runners: dict[str, Runner] = {
        "SkTree": SkitTreeRunner,
        "TfdfTree": TfdfRunner,
    }

    # Parse Args
    parser = arg_parser()  # Use the standard runner args
    subparsers = parser.add_subparsers(dest="runner")

    for k, v in runners.items():
        sp = subparsers.add_parser(k)
        v.arg_parser(sp)
        # sp.set_defaults(v)

    args = parser.parse_args()
    print(args)

    # set_gpu_growth()
    # split_gpus(num_vgpus=3)
    # tf.debugging.set_log_device_placement(True)

    logger.info(f"Tensorflow Version: {tf.__version__}")
    # Equivalent to the two lines above
    # logger.info(f"Setting compute mode to 'mixed_float16'")
    # tf.keras.mixed_precision.set_global_policy('mixed_float16')
    # tf.config.experimental.enable_tensor_float_32_execution(False)
    hp = HP()
    hp.Fixed("BATCH_SIZE", 512)
    hp.Fixed("NUM_EPOCHS", 20)
    hp.Fixed("USE_WEATHER", True)
    hp.Fixed("WARMUP_STEPS", 24)  # used as a slices so will take e.g [0, 1, ..., 23]
    hp.Fixed("OUTPUT_STEPS", 24)
    hp.Fixed("OUTPUT_STEPS_start", hp.get("WARMUP_STEPS") - 25 + 12)  # these are offset by 24 in the data read pipeline
    hp.Fixed("OUTPUT_STEPS_end", hp.get("WARMUP_STEPS") + hp.get("OUTPUT_STEPS") - 24 + 12)  # these are offset by 24 in the data read pipeline
    hp.Config("Test", 5)

    runner = runners[args.runner](hp=hp, **args.__dict__)
    runner.run()
