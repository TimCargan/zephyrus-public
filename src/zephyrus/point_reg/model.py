import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras import layers
import data_pipline


def build_lstm_model(hp: kt.HyperParameters) -> tf.keras.Model:
    fs = ['Longitude', 'Latitude', 'clouds', 'pres', 'precip', 'temp', 'rh', 'ghi', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos', 'wind_x', 'wind_y']
    inputs_ts = [layers.Input(shape=(120),  name=f) for f in fs]

    i_irrad = layers.Input(shape=(120),  name="irradiance")
    irrad_24 = i_irrad[:,:24,]
    # layers.ZeroPadding1D #[[0, 0], [padding[0], padding[1]], [0, 0]]
    irrad = tf.pad(irrad_24, [[0,0], [0,96]], constant_values=-1)



    # [Batch, Time (120), Feats]
    x = tf.stack(inputs_ts + [irrad] , axis=-1)
    x = x[:, :24, ]
    x = layers.BatchNormalization()(x)


    x = layers.Bidirectional(layers.LSTM(units=128, return_sequences=True))(x)

    # [Batch, Time (72), Feats]
    x = layers.BatchNormalization()(x)
    x = x[:, :24, ]

    x = layers.Bidirectional(layers.LSTM(units=64, return_sequences=True))(x)

    x = layers.Dense(1, dtype='float32')(x)
    # x = layers.ReLU(dtype='float32')(x)
    x = tf.squeeze(x)
    x = x + irrad_24

    inputs_ts.append(i_irrad)
    return tf.keras.Model(inputs=inputs_ts , outputs=x)

def build_1dc_model(hp:kt.HyperParameters) -> tf.keras.Model:
    fs = ["pres", "precip", "clouds", "wind_spd", "wind_dir", "temp", "rh", "ghi"]
    static_feats = ["DOY", "Hour", "Longitude", "Latitude"]

    inputs_ts = [layers.Input(shape=(120),  name=f) for f in fs]
    inputs_static = [layers.Input(shape=(1), name=f) for f in static_feats]

    x = tf.stack(inputs_ts, axis=-1)
    x = layers.Conv1D(32, 7, strides=3, activation="ReLU")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(hp.get("OUTPUT_STEPS"))(x)
    outputs = layers.ReLU()(x)

    model = tf.keras.Model(inputs=inputs_ts, outputs=outputs)
    return model


def build_model(hp:kt.HyperParameters) -> tf.keras.Model:
    fs = ['Longitude', 'Latitude', 'clouds', 'pres', 'precip', 'temp', 'rh', 'ghi', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos', 'wind_x', 'wind_y']

    #fs = ["pres", "precip", "clouds", "wind_spd", "wind_dir", "temp", "rh", "ghi"]
    static_feats =[] # ["DOY", "Hour", "Longitude", "Latitude" ]

    inputs = [layers.Input(shape=(120),  name=f) for f in fs]
    inputs += [layers.Input(shape=(1), name=f) for f in static_feats]

    x = tf.concat(inputs, axis=-1)
    size = 256 #hp.Choice(f"width", [72, 128, 256])
    for i in range(hp.Int("depth", min_value=2, max_value=4)):
        if size is not None:
            x = layers.Dense(size, activation="relu", name=f"layer_{i}")(x)
            x = layers.Dropout(0.3)(x)


    x = layers.Dense(hp.get("OUTPUT_STEPS"))(x)
    outputs = layers.ReLU(dtype='float32')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model