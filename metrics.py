import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import MeanSquaredError

# Define an RMSE metric
def root_mean_squared_error(y_true, y_pred):

    #y_true = tf.cast(y_true, dtype=tf.float32)
    #y_pred = tf.cast(y_pred, dtype=tf.float32)


    #return K.sqrt(K.mean(K.square(y_pred - y_true)) + K.epsilon())
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)) + tf.keras.backend.epsilon())