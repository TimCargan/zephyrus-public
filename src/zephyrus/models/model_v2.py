import keras_tuner as kt
import math
import tensorflow as tf
from tensorflow.keras import layers

from zephyrus.models.layers import MultiHead, ImputationStep, DenseLayers
from zephyrus.models.util import array_get
from zephyrus.models.loss import ZeroLoss


def build_keras_model(hp: kt.HyperParameters) -> tf.keras.Model:
    model = build_model(hp)
    model.compile(
        loss={
            "y_hat": "MSE",
            # "y_pi": PI_loss()
        },
        loss_weights={
            "y_hat": 1,
            # "y_pi": 1
        },
        metrics={
            "y_hat": [ZeroLoss(step=0, name="t0"), ZeroLoss(step=1, name="t1"), ZeroLoss(step=-1, name="te")]}
    )
    return model


class ConvoltionModule(layers.Layer):
    """
    This is the ConvoltionModule image layer
    :param hp: Hyper parameters
    :return: (inputs, x), input layers and output value
    """
    def __init__(self, hp: kt.HyperParameters, im_inputs, **kwargs):
        super().__init__(**kwargs)
        self.num_conv_heads = hp.Int("NUM_CONV_HEADS", min_value=2, max_value=3, default=4)
        self.kernel_size = hp.Int("HEAD_K_SIZE", min_value=3, max_value=5, default=3)
        self.filters = hp.Int("HEAD_FILTERS", min_value=2, max_value=4, default=3)

        self.heads = MultiHead(len(im_inputs),
                               num_convs=self.num_conv_heads,
                               kernel_size=self.kernel_size,
                               filters=self.filters,
                               name="MultiConvHeads")

    def call(self, x, **kwargs):
        x = self.heads(x)
        return x


class ImputeModule(layers.Layer):
    def __init__(self, hp: kt.HyperParameters, **kwargs):
        super().__init__(**kwargs)
        self.batch_norm = hp.Boolean("PRE_IMPUTE_BATCH_NORM", False)

        self.stateful = hp.Fixed("Statefull", False)
        self.LEAD_STEPS = hp.get("LEAD_STEPS")

        self.BIDIRECTIONAL = hp.Boolean("BIDIRECTIONAL", True)
        self.IMPUTE_CHANNELS = hp.Choice("IMPUTE_CHANNELS", [1,2], default=1)
        self.LSTM_REV = hp.get("LSTM_REV")
        self.NUM_CONV_LSTMS = hp.Int("TOTAL_CONVS", min_value=3, max_value=5, default=3) - hp.get("NUM_CONV_HEADS")

        self.PLSTM_KERNEL_SIZE = hp.Int("PLSTM_K_SIZE", min_value=2, max_value=4, default=3)
        self.IMPUTER_KERNEL_SIZE = hp.Int("IMPUTER_K_SIZE", min_value=3, max_value=5, default=3)
        self.PLSTM_FILTERS = hp.Choice("PLSTM_FILTERS", [4])

        if self.batch_norm:
            self.norm = layers.BatchNormalization()

        self.imputer = ImputationStep(self.LEAD_STEPS,
                                      stateful=self.stateful,
                                      bidirectional=self.BIDIRECTIONAL,
                                      plstm_layers=self.NUM_CONV_LSTMS,
                                      impute_channels=self.IMPUTE_CHANNELS,
                                      go_backwards=self.LSTM_REV,
                                      plstm_kernel_size=self.PLSTM_KERNEL_SIZE,
                                      imputer_kernel_size=self.IMPUTER_KERNEL_SIZE,
                                      plstm_filters=self.PLSTM_FILTERS)

    def call(self, x, **kwargs):
        if self.batch_norm:
            x = self.norm(x)
        x = self.imputer(x)
        return x


class RegModule(layers.Layer):
    def __init__(self, hp: kt.HyperParameters, **kwargs):
        super().__init__(**kwargs)
        NUM_DENSE_LAYERS: int = hp.Int("NUM_DENSE_LAYERS", min_value=1, max_value=3, default=1)
        REG_MAX_SIZE = hp.Choice("REG_MAX_SIZE", [64, 128, 256], default=128)
        REG_SHAPE = hp.Choice("REG_SHAPE", ["SQUARE", "FUNNEL"], default="FUNNEL")
        FUNNEL_FACTOR = hp.Choice("FUNNEL_FACTOR", [2, 8], default=8, parent_name="REG_SHAPE", parent_values=["FUNNEL"])

        if REG_SHAPE == "FUNNEL":
            max_size = REG_MAX_SIZE
            c_size = max_size
            REG_MAX_SIZE = []
            for i in range(NUM_DENSE_LAYERS):
                layer_size = max(1, math.floor(c_size))
                REG_MAX_SIZE.append(layer_size)
                c_size = c_size / FUNNEL_FACTOR

        self.reg = layers.TimeDistributed(DenseLayers(num_layers=NUM_DENSE_LAYERS, shape=REG_MAX_SIZE), name=f"meta_mix")

    def call(self, x, **kwargs):
        return self.reg(x)


def embed(embeddings, em_size):
    """
    Convert a zipped list of inputs and info into embeddings
    TODO: make this a layer somehow?
    :param embeddings: list[tuple(input_layer, embed_info)]
    :return: list of embedded value tensors
    """
    def _embed(em, info):
        (name, num_keys) = info
        em = layers.TimeDistributed(layers.Embedding(num_keys, em_size), name=f"embed_{name}")(em)
        return layers.TimeDistributed(layers.Flatten(), name=f"flatten_{name}")(em)
    return [_embed(*x) for x in embeddings]


def load_meta_data(hp):
    """
    Load and embed the metadata
    :param hp:
    :return: [meta] - A list of tensors tensor with metadata
    """
    LEAD_STEPS = hp.get("LEAD_STEPS")
    LOCAL_BATCH_SIZE = hp.get("LOCAL_BATCH_SIZE")
    LOCAL_BATCH_SIZE = LOCAL_BATCH_SIZE if LOCAL_BATCH_SIZE > 0 else None #Hack
    BATCH_SIZE = hp.get("BATCH_SIZE")
    TIME_STEPS = hp.get("TIME_STEPS")
    em_size = LEAD_STEPS
    em_in_shape = (em_size, 1,)


    EMBEDDING_LAYERS = [("lat", 500), ("lon", 500)]
    META_LAYERS = ['azimuth', 'zenith', 'elevation', 'equation_of_time', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos']
    # need to hold a reference to the input layers
    meta_inputs = [layers.Input(name=n, shape=em_in_shape, dtype=tf.float16) for n in META_LAYERS]
    em_inputs =   [layers.Input(name=n, shape=em_in_shape) for n, _ in EMBEDDING_LAYERS]
    embeddings = embed(zip(em_inputs, EMBEDDING_LAYERS), hp.Int("EMBEDDING_SIZE", 5, 13))

    #  Adds index, i.e distance from t_0 as meta input
    # TODO: Clean this up a bit

    dist = tf.range(0, LEAD_STEPS, dtype=tf.float32)
    dist = tf.reshape(dist, (1, LEAD_STEPS, 1))
    batch_ss = LOCAL_BATCH_SIZE if LOCAL_BATCH_SIZE else BATCH_SIZE # Batch tile size
    dist = tf.tile(dist, [batch_ss, 1, 1])

    meta = embeddings + meta_inputs
    # meta.append(dist)

    return meta, em_inputs + meta_inputs


def build_model(hp: kt.HyperParameters) -> tf.keras.Model:

    #####################
    # Convoltion Module
    #####################
    IM_LAYERS = array_get(hp, "IM_LAYERS")  # TODO Find a nice way to pass this in

    im_shape = (hp.get("TIME_STEPS"), hp.get("IM_HIGHT"), hp.get("IM_WIDTH"), hp.get("IM_CHANNEL"))
    im_inputs = [layers.Input(name=n, shape=im_shape) for n in IM_LAYERS]

    x = tf.stack(im_inputs, axis=2, name="StackImages")
    x = ConvoltionModule(hp, IM_LAYERS)(x)

    #####################
    # Imputation Module
    #####################
    # x = ImputeModule(hp)(x)
    x = layers.TimeDistributed(layers.Flatten(), name="FlattenImputation")(x)
    #####################
    # Metadata loading
    #####################
    meta, meta_em_inputs = load_meta_data(hp)
    past_irrad = layers.Input(name="past_irrad", shape=(hp.get("LEAD_STEPS"), 12, ), dtype=tf.float16)
    step = layers.Input(name="step", shape=(hp.get("LEAD_STEPS"), 1), dtype=tf.float16)
    meta.append(step)
    meta.append(past_irrad)
    meta.append(x)
    x = tf.concat(meta, axis=2)
    #####################
    # Regression module
    #####################
    if hp.Fixed("PREREG_BATCH_NORM", True):
        x = layers.BatchNormalization()(x)

    x = RegModule(hp)(x)
    # Final Output Layers
    y_hat = layers.TimeDistributed(layers.Dense(1, activation='relu', dtype=tf.float32),  name="y_hat")(x)
    # y_hat = tf.squeeze(y_hat, name="y_hat")
    y_pi = layers.TimeDistributed(layers.Dense(2, activation='relu'), name="y_pi")(x)

    # TODO: THis is a very hacky way to do this, should clean up
    if hp.get("META_ONLY"):
        inputs = meta_em_inputs
    else:
        inputs = im_inputs + meta_em_inputs + [past_irrad, step]

    outputs = {"y_hat": y_hat} #y_pi
    if hp.get("OUTPUT_PASSTHROUGH"):
        # inputs / output for faster post processing
        meta_data = {"plant": layers.Input(name="plant", shape=(1), dtype=tf.string),
                     "tz": layers.Input(name="tz", shape=(hp.get("LEAD_STEPS"))),
                     "y": layers.Input(name="y", shape=(hp.get("LEAD_STEPS")))}

        inputs += meta_data.values()
        outputs = {**outputs, **meta_data, "step": tf.cast(step, dtype=tf.int32)}

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


