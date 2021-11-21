import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras import layers


class FeedBack(layers.Layer):
    def __init__(self, hps, units, out_steps):
        super().__init__()
        self.hps = hps
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(1)
        self.dense_mix = tf.keras.layers.Dense(32)
        self.dense_pre = tf.keras.layers.Dense(32)
        self.bn = tf.keras.layers.BatchNormalization()

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x = tf.keras.layers.TimeDistributed(self.dense_pre)(inputs)
        x, *state = self.lstm_rnn(x)

        # predictions.shape => (batch, features)
        x = self.dense_mix(x)
        prediction = self.dense(x)
        return prediction, state


    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the lstm state
        irrad_weather_warm, weather = inputs

        prediction, state = self.warmup(irrad_weather_warm)

        # Insert the first prediction
        predictions.append(prediction)

        # Run the rest of the prediction steps
        for n in range(0, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            weather_n = tf.squeeze(weather[:,n,])
            x = tf.concat([weather_n, x], axis=-1)
            # Execute one lstm step.
            x = self.dense_pre(x)
            x, state = self.lstm_cell(x, states=state,
                                      training=training)
            # Convert the lstm output to a prediction.
            x = self.dense_mix(x)

            prediction = self.dense(x)
            # Add the prediction to the output
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions

class FeedBack_no_weather(FeedBack):
    def __init__(self, hps, units, out_steps):
        super().__init__(hps, units, out_steps)


    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the lstm state
        prediction, state = self.warmup(inputs)

        # Insert the first prediction
        predictions.append(prediction)

        # Run the rest of the prediction steps
        for n in range(0, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x = self.dense_pre(x)
            x, state = self.lstm_cell(x, states=state,
                                      training=training)
            # Convert the lstm output to a prediction.
            x = self.dense_mix(x)

            prediction = self.dense(x)
            # Add the prediction to the output
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions

def build_auto_reg(hp: kt.HyperParameters) -> tf.keras.Model:
    fs = ['Longitude', 'Latitude', 'clouds', 'pres', 'precip', 'temp', 'rh', 'ghi', 'hour_sin', 'hour_cos', 'year_sin', 'year_cos', 'wind_x', 'wind_y']
    inputs_ts = [layers.Input(shape=(120),  name=f) for f in fs]

    warm_up_steps = hp.get("WARMUP_STEPS")
    i_irrad = layers.Input(shape=(120),  name="irradiance")
    irrad_24 = i_irrad[:,12:warm_up_steps + 12,]

    weather = tf.stack(inputs_ts, axis=-1)

    # This groups together a slice of 24 timsteps into one so we can feed the lstm past and futuer weather states in a single step
    v = []
    sl = 24
    for i in range(0, 65):
        s = weather[:, i: i + sl, :]
        s = tf.reshape(s, [-1, sl * 14])
        v.append(s)
    weather = tf.stack(v, axis=1)

    # first 24 weather points and add irrad as a feat
    weather_irrad = tf.concat([weather[:, :warm_up_steps,], tf.expand_dims(irrad_24, -1)], axis=-1) #[:,22:,]#use this to cut down the warmup to the last x hours


    if hp.get("USE_WEATHER"):
        x = FeedBack(hp, units=128, out_steps=hp.get("OUTPUT_STEPS"))((weather_irrad, weather[:, warm_up_steps:, ]))
    else:
        x = FeedBack_no_weather(hp, units=128, out_steps=hp.get("OUTPUT_STEPS"))(tf.expand_dims(irrad_24, -1))

    x = tf.squeeze(x)
    x = tf.cast(x, dtype=tf.float32)
    inputs_ts.append(i_irrad)
    return tf.keras.Model(inputs=inputs_ts , outputs=x)

