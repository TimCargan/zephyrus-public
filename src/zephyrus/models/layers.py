import math

import tensorflow as tf
from tensorflow.keras import layers


class ConvHead(layers.Layer):
    def __init__(self, num_convs=2, filters=4, kernel_size=3, data_format='channels_last', **kwargs):
        super(ConvHead, self).__init__(**kwargs)
        self.conv_name = kwargs.get('name')
        self.num_convs = num_convs
        self.convs = None
        self.data_format = data_format
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        if self.built:
            return
        self.built = True

        self.convs = []
        self.bns = []
        for i in range(self.num_convs):
            conv = layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=(1, 1), activation=None,
                                 data_format=self.data_format, name=f"{self.conv_name}_conv_{i}")
            self.convs.append(conv)
            self.bns.append(layers.BatchNormalization())

    def compute_output_shape(self, input_shape):
        batch, x, y, c = input_shape
        nx, ny = x, y
        for _ in range(self.num_convs):
            nx = math.floor((nx - 2) / 2) + 1
            ny = math.floor((ny - 2) / 2) + 1
        return (batch, nx, ny, self.filters)

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        for i in range(self.num_convs):
            conv = self.convs[i]
            x = layers.ZeroPadding2D(1, name=f"{self.conv_name}_pad_{i}")(x)
            x = conv(x)
            x = self.bns[i](x)
            x = layers.ReLU()(x)
            x = layers.MaxPool2D(2, name=f"{self.conv_name}_maxpool_{i}")(x)

        return x

    def get_config(self):
        config = super(ConvHead, self).get_config()
        config.update({'num_convs': self.num_convs, 'data_format': self.data_format})
        return config


class MultiHead(layers.Layer):
    def __init__(self, num_heads, num_convs=2, filters=4, kernel_size=3, names=None, reutrn_list=False, data_format='channels_last', **kwargs):
        super(MultiHead, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.num_convs = num_convs
        self.heads = None
        self.data_format = data_format
        self.reutrn_list = reutrn_list
        self.filters = filters
        self.kernel_size = kernel_size

        self.names = names
        if self.names is not None:
            assert len(names) == num_heads

    def build(self, input_shape):
        if self.built:
            return
        self.built = True

        self.heads = []
        for i in range(self.num_heads):
            name = None
            if self.names is not None:
                name = self.names[i]
            head = ConvHead(self.num_convs, filters=self.filters, kernel_size=self.kernel_size, name=name, data_format=self.data_format)
            self.heads.append(head)

    # @tf.function
    def call(self, inputs, **kwargs):
        res = []
        for i, head in enumerate(self.heads):
            x = layers.TimeDistributed(head)(inputs[:,:, i])
            res.append(x)

        if self.reutrn_list:
            return res

        return layers.concatenate(res)

    def get_config(self):
        config = super(MultiHead, self).get_config()
        config.update({'num_heads': self.num_heads, 'num_convs': self.num_convs, 'names': self.names,
                       'data_format': self.data_format})
        return config

    # def compute_output_shape(self, input_shape):
    #     return  (None, 16,16, 4)


class DenseLayers(layers.Layer):
    def __init__(self, num_layers=None, shape=None, activation='relu', dropout=True, dropout_rate=0.2, lname="", **kwargs):
        super(DenseLayers, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.layers = []
        self.lname = lname

        if isinstance(shape, list):
            self._shapes = shape
            self.num_layers = len(shape)
        else:
            self._shapes = [shape] * num_layers

        if isinstance(activation, list):
            assert len(activation) == self.num_layers
            self._activatons = activation
        else:
            self._activatons = [activation] * self.num_layers

    def build(self, input_shape):
        for i in range(len(self._shapes)):
            l = layers.Dense((self._shapes[i]), activation=self._activatons[i])  # Name layers
            self.layers.append(l)

            if self.dropout:
                l = layers.Dropout(self.dropout_rate)
                self.layers.append(l)

    def compute_output_shape(self, input_shapes):
        batch, x = input_shapes
        return (batch, self._shapes[-1])

    @tf.function
    def call(self, inputs, **kwargs):
        x = inputs
        # TODO: this is a fold...
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class LSTMLayers(DenseLayers):
    def __init__(self, num_layers=None, shape=None, activation='relu', dropout=True, dropout_rate=0.2, lname="", **kwargs):
        super(LSTMLayers, self).__init__(num_layers, shape, activation,
                                         dropout, dropout_rate, lname="", **kwargs)

    def build(self, input_shape):
        for i in range(self.num_layers):
            l = layers.LSTM((self._shapes[i]), return_sequences=True)  #TODO: Name layers
            self.layers.append(l)

            if self.dropout:
                l = layers.Dropout(self.dropout_rate)
                self.layers.append(l)


class GRULayers(DenseLayers):
    def __init__(self, num_layers=None, shape=None, activation='relu', dropout=True, dropout_rate=0.2, lname="", **kwargs):
        super(GRULayers, self).__init__(num_layers, shape, activation,
                                         dropout, dropout_rate, lname="", **kwargs)

    def build(self, input_shape):
        for i in range(self.num_layers):
            l = layers.GRU((self._shapes[i]), return_sequences=True)  #TODO: Name layers
            self.layers.append(l)

            if self.dropout:
                l = layers.Dropout(self.dropout_rate)
                self.layers.append(l)


class MixLayers(DenseLayers):
    def __init__(self, num_layers=None, shape=None, activation='relu', dropout=True, dropout_rate=0.2, lname="", **kwargs):
        super(MixLayers, self).__init__(num_layers, shape, activation,
                                         dropout, dropout_rate, lname="", **kwargs)

    def _get_layer(self, layer_type, **kwargs):
        l = None
        if layer_type == "Dense":
            l = layers.TimeDistributed(layers.Dense(**kwargs))
        elif layer_type == "LSTM":
            l = layers.GRU(**kwargs)
        elif layer_type == "GRU":
            l = layers.GRU(**kwargs)

        assert l
        return l

    def build(self, input_shape):
        for i in range(self.num_layers):
            l = layers.GRU((self._shapes[i]), return_sequences=True)  # TODO: Name layers
            self.layers.append(l)

            if self.dropout:
                l = layers.Dropout(self.dropout_rate)
                self.layers.append(l)



class Imputation(layers.Layer):
    """ Imputation Layer
    Converts between data of varying frequencies
    """

    def __init__(self, output_rate, input_rate=None, plstm_layers=1, data_format="channels_last", stateful=False,
                 return_sequences=True, bidirectional=False, go_backwards=False, impute_channels=1,
                 plstm_kernel_size=3, imputer_kernel_size=5, plstm_filters=16, **kwargs):
        super(Imputation, self).__init__(**kwargs)
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.data_format = data_format
        self.plstm_layers = plstm_layers
        self.stateful = stateful
        self.return_sequences = return_sequences
        self.bidirectional = bidirectional
        self.impute_channels = impute_channels
        self.go_backwards = go_backwards
        self.pre_impute_layers = None
        self.imputer = None
        self.plstm_kernel_size = plstm_kernel_size
        self.plstm_filters = plstm_filters
        self.imputer_kernel_size = imputer_kernel_size

    def build(self, input_shape):
        if self.built:
            return
        self.built = True

        if self.input_rate is None:
            self.input_rate = input_shape[1]
        else:
            assert self.input_rate == input_shape[1]

        self.pre_impute_layers = []
        for i in range(self.plstm_layers):
            pl = layers.ConvLSTM2D(self.plstm_filters, kernel_size=self.plstm_kernel_size, strides=(1, 1), stateful=self.stateful, padding="same",
                                   data_format=self.data_format, return_sequences=True, name=f"conv_lstm_{i}",
                                   go_backwards=self.go_backwards)
            if self.bidirectional:
                pl = layers.Bidirectional(pl, merge_mode='sum')
            self.pre_impute_layers.append(pl)

        output_els = self.output_rate * self.impute_channels
        self.imputer = layers.ConvLSTM2D(output_els, kernel_size=self.imputer_kernel_size, strides=(1, 1), stateful=self.stateful, padding="same",
                                         data_format=self.data_format, return_sequences=True, name="conv_lstm_impute",
                                         go_backwards=self.go_backwards)
        if self.bidirectional:
            self.imputer = layers.Bidirectional(self.imputer, merge_mode='sum')

    def call(self, inputs):
        x = inputs
        for i, layer in enumerate(self.pre_impute_layers):
            x = layer(x)
            x = layers.TimeDistributed(layers.MaxPool2D(2), name="max_pool")(x)
            x = layers.TimeDistributed(layers.ZeroPadding2D(2), name="lstm_pad")(x)

        x = self.imputer(x)

        # Move imputed channels to the front to combine with time
        if self.data_format == "channels_last":
            x = layers.Permute((1, 4, 2, 3))(x)

        # Reshape to combine output channels with the time dimension
        x = layers.Reshape((self.input_rate * self.output_rate, self.impute_channels, *x.shape[3:]))(x)

        # Move image channels to end
        if self.data_format == "channels_last":
            x = layers.Permute((1, 3, 4, 2))(x)

        if not self.return_sequences:
            x = x[:, -self.output_rate:, ]

        return x

    def reset_states(self):
        if self.stateful:
            for lstm in self.pre_impute_layers:
                lstm.reset_states()
            self.imputer.reset_states()

    def get_config(self):
        config = super(Imputation, self).get_config()
        config.update({'input_rate': self.input_rate, 'output_rate': self.output_rate,
                       'data_format': self.data_format, 'plstm_layers': self.plstm_layers,
                       'stateful': self.stateful, 'return_sequences': self.return_sequences})
        return config


class ImputationStep(Imputation):

    def __init__(self, output_rate, scale_reduce=True, **kwargs):
        super(ImputationStep, self).__init__(output_rate, **kwargs)
        self.scale_reduce = scale_reduce

    def build(self, input_shape):
        super(ImputationStep, self).build(input_shape)
        x = 1  # input_shape[2] // 2**self.plstm_layers
        y = 1  # input_shape[3] // 2**self.plstm_layers
        c = 1
        scale_shape = [self.input_rate, x, y, self.output_rate, c]
        if self.scale_reduce:
            self.output_scale = self.add_weight(shape=scale_shape, name=f"scale_dim", initializer='glorot_uniform')

    def call(self, inputs):
        x = inputs

        for i, layer in enumerate(self.pre_impute_layers):
            x = layer(x)
            x = layers.TimeDistributed(layers.MaxPool2D(2), name="max_pool")(x)
            x = layers.TimeDistributed(layers.ZeroPadding2D(2), name="lstm_pad")(x)

        x = self.imputer(x)

        # Reshape to combine output channels with the time dimension
        x = layers.Reshape((self.input_rate, *x.shape[2:4], self.output_rate, self.impute_channels,))(x)
        # Scale each time-step
        if self.scale_reduce:
            x = tf.math.multiply(x, self.output_scale)

        # Reduce scaled time-steps on input axis i.e [in, out, x,y,c] -> [out, x, y, c]
        # inpt, x, y, otpt, c = 1, 2, 3, 4, 5
        # [in, x, y, out, c] -> [out, x, y, c, in]
        x = tf.transpose(x, [0, 4, 2, 3, 5, 1], name="reorder")  # layers.Permute((1,4,2,3,5))(x)
        x = tf.math.reduce_sum(x, axis=5)  # TODO: review how this happens

        # # Move imputed channels to the front to combine with time
        # if self.data_format == "channels_last":
        #     x = layers.Permute((1, 4, 2, 3))(x)
        # # # Reshape to combine output channels with the time dimension
        # # x = layers.Reshape((self.input_rate, self.output_rate, self.impute_channels, *x.shape[3:]))(x)
        # # # Scale each time-step
        # # if self.scale_reduce:
        # #     x = tf.math.multiply(x, self.output_scale)
        # # # Reduce scaled time-steps on input axis i.e [in, out, x,y,c] -> [out, x, y, c]
        # # x = layers.Permute((2, 3, 4, 5, 1))(x)
        # # x = tf.math.reduce_sum(x, axis=5) #TODO: review how this happens
        # # Move image channels to end
        # if self.data_format == "channels_last":
        #     x = layers.Permute((1, 3, 4, 2))(x)

        if not self.return_sequences:
            x = x[:, -self.output_rate:, ]

        return x


class ImputationConv(Imputation):
    """ Imputation Layer
    Hack for no impute tests to keep images the same size
    """
    def __init__(self, output_rate, **kwargs):
        super(ImputationConv, self).__init__(output_rate, **kwargs)

    def build(self, input_shape):
        if self.built:
            return
        self.built = True

        if self.input_rate is None:
            self.input_rate = input_shape[1]
        else:
            assert self.input_rate == input_shape[1]

        self.pre_impute_layers = []
        for i in range(self.plstm_layers):
            pl = layers.Conv2D(16, kernel_size=3, strides=(1, 1), padding="same",
                               data_format=self.data_format, name=f"conv_lstm_{i}")
            self.pre_impute_layers.append(pl)

        output_els = self.output_rate * self.impute_channels
        self.imputer = layers.Conv2D(output_els, kernel_size=5, strides=(1, 1),
                                     data_format=self.data_format, name="conv_lstm_impute")

    @tf.function
    def call(self, inputs):
        x = inputs
        for i, layer in enumerate(self.pre_impute_layers):
            x = layers.TimeDistributed(layer)(x)
            x = layers.TimeDistributed(layers.MaxPool2D(2), name="max_pool")(x)
            x = layers.TimeDistributed(layers.ZeroPadding2D(2), name="lstm_pad")(x)

        x = layers.TimeDistributed(self.imputer)(x)

        # Move imputed channels to the front to combine with time
        if self.data_format == "channels_last":
            x = layers.Permute((1, 4, 2, 3))(x)

        # Calculate the final size of the imputed data that is flattened out
        x = layers.Reshape((self.input_rate * self.output_rate, self.impute_channels, *x.shape[3:]))(x)

        # Move image channels to end
        if self.data_format == "channels_last":
            x = layers.Permute((1, 3, 4, 2))(x)

        if not self.return_sequences:
            x = x[:, -self.output_rate:, ]

        return x
