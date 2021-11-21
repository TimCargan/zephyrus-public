import tensorflow as tf

# https://github.com/TeaPearce/Deep_Learning_Prediction_Intervals/blob/master/code/DeepNetPI.py
class PI_loss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.1, s=160.0, lambda_=10.0, epsilon=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.s = s
        self.lambda_ = lambda_
        self.epsilon = epsilon

    def call(self, y_true, y_pred, sample_weight=None):
        """
        :param y_true: shape = [batch_size, d0, .. dN]
        :param y_pred: shape = [batch_size, d0, .. dN]
        :param sample_weight:
        :return:
        """

        # Extract and rename inputs
        y = y_true
        # [Batch, Timestems, du/dl]
        yu,yl = y_pred[:, :, 0], y_pred[:, :, 1]

        loss, metrics = self.loss(y, yu, yl, self.alpha, self.s, self.lambda_, self.epsilon)
        return loss

    @staticmethod
    def loss(y, yu, yl, alpha=0.1, s=160.0, lambda_=10.0, epsilon=1e-4):
        mpiw_real = tf.reduce_mean(tf.subtract(yu, yl))

        # I think this might be causing the issue, since if y = yu = yl it considers it to be outside of the range
        khu = tf.math.maximum(0.0, tf.sign(1 + (2 * tf.sign(yu - y))))  # make zero + 1
        khl = tf.math.maximum(0.0, tf.sign(1 + (2 * tf.sign(y - yl))))
        kh = khu * khl

        # Shouldn't need the nudge to push them appart since sigmode(0) = 0.5 so unless the delta is <0.0001 should be ok
        ksu = tf.math.sigmoid((yu - y) * s)
        ksl = tf.math.sigmoid((y - yl) * s)
        ks = ksu * ksl

        # In the algo on the paper this term is referred to as mpiw_capt
        # (MPIW captured, i.e only count values have the correct width)
        mpiw_hard = tf.math.reduce_sum(tf.abs(yu - yl) * kh) / (tf.math.reduce_sum(kh) + epsilon)
        mpiw_soft = tf.math.reduce_sum(tf.abs(yu - yl) * ks) / (tf.math.reduce_sum(ks) + epsilon)

        picp_hard = tf.reduce_mean(kh)
        picp_soft = tf.reduce_mean(ks)

        n = tf.cast(tf.size(y), "float32") # Sample size
        loss_rhs = tf.sqrt(n) * tf.square(tf.maximum(0.0, (1.0 - alpha) - picp_soft))

        loss = mpiw_soft + lambda_ * loss_rhs

        return loss, (mpiw_real, picp_hard, loss_rhs, mpiw_soft, picp_soft)



class VecLoss(tf.keras.metrics.RootMeanSquaredError):
    def __init__(self, name = 'vec_error', dtype = None):
        super(VecLoss, self).__init__(name, dtype = dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.mean = tf.reduce_mean(y_pred - y_true, 0)
        return super(VecLoss, self).update_state(y_true, y_pred)

    def result(self):
        return self.mean


# Lazy hack to so i can add this as a test metric
class ZeroLoss(tf.keras.metrics.RootMeanSquaredError):
    def __init__(self, step=0, name = 'mean_absolute_error', dtype = None):
        super(ZeroLoss, self).__init__(name, dtype = dtype)
        self.z_index = step #(hps.TIME_STEPS - 1) * hps.LEAD_STEPS

    def update_state(self, y_true, y_pred, sample_weight=None):
        z_index = self.z_index
        zy_true = y_true[:, z_index,]
        zy_pred = y_pred[:, z_index,]
        return super(ZeroLoss, self).update_state(zy_true, zy_pred, sample_weight)


# Lazy hack to so i can add this as a test metric
class SSIMLoss(tf.keras.metrics.Mean):
    def __init__(self, hps, name = 'ssim_loss', dtype = None):
        super(SSIMLoss, self).__init__(name, dtype=dtype)

    def update_state(self, y_true: tf.Tensor, y_pred, sample_weight=None):
        simm = tf.image.ssim(y_true, y_pred, max_val=1.0)
        super(SSIMLoss, self).update_state(simm, sample_weight)