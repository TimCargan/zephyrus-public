import math

import numpy as np
import tensorflow as tf
from src.zephyrus.data_pipelines.tile_utils import tilesUtils

# Constants
DAY_SECONDS = 24 * 60 * 60  # seconds in a day
YEAR_SECONDS = 365.2425 * DAY_SECONDS  # seconds in a day year

def scale_cord(cord, o_range, t_range):
    """
    Scale from cords with variable range to a 0 ... t_range
    :param cord: value to convert
    :param o_range: a tuple (min, max) of cord range
    :param t_range: max value in zero based target range
    :return: Scaled cord value
    """
    _range = o_range[1] - o_range[0]
    scale = t_range / _range
    s_cord = (cord - o_range[0]) * scale
    return int(s_cord)


def lat_long_to_onehot_grid(lat, long,height=128,width=128,lat_range=(-14,3),long_range=(49,62)):
    """
    Create a 2D tensor with all zeros except for the cell corresponding to the given lat long
    :param lat: latitude (Lat -> Height)
    :param long: longitude (Long -> Width)
    :param height: output height
    :param width: output width
    :param lat_range: a tuple (min, max) of the range of latitude values
    :param long_range: a tuple (min, max) of the range of longitude values
    :return: tensors (height,width) of zeors where lat,long is 1
    """
    long_c = scale_cord(long, long_range, width)
    lat_c = scale_cord(lat, lat_range, height)
    return tf.scatter_nd([[lat_c,long_c]], [1], [height,width])


def latlonToTileId(plant):
    lat,lon = plant["Latitude"], plant["Longitude"]
    tiles7 = tilesUtils(7)
    pix = tiles7.fromLatLngToTilePixel(lat, lon, 7)
    area = {"tile_x": pix.tx, "tile_y" :pix.ty, "px_x": pix.px, "px_y" :pix.py}
    x = int((area["tile_x"] - 59) * 256 + area["px_x"])
    y = int((area["tile_y"] - 36) * 256 + area["px_y"])
    return x,y


def latlongtoimage(lat, lon, img=(500,500), rng=((61, 48), (-12,5))):
    """
    Lat is N/S 61,48,
    Lon is E/W -12, 5
    :param plant:
    :param img: a tuple of the image size in the form (height, width)
    :param rng: a tuple of area the image covers in the form ((N, S), (W,E))
    :return: a tuple (x,y) for pixel that is plant
    """
    def scale(point, source, size):
        source_range = source[1] - source[0]
        conv_p = ((point - source[0]) / source_range) * size
        return conv_p

    new_y = int(scale(lat, rng[0], img[0]))
    new_x = int(scale(lon, rng[1], img[1]))

    return new_x, new_y


def vectorize(speed, dir_deg=None, dir_rad=None):
    """
    Convert as speed and direction into its component vectors
    Args:
        speed: magnitude of vector
        dir_deg: direction in degrees
        dir_rad: direction in radians

    Returns: A tuple of (x_component, y_component)
    """
    # Convert to radians.
    if dir_rad is None:
        dir_deg = tf.cast(dir_deg, tf.float32)
        dir_rad = dir_deg * math.pi / 180

    speed = tf.cast(speed, tf.float32)
    # Calculate the x and y components.
    x_comp = speed * tf.math.cos(dir_rad)
    y_comp = speed * tf.math.sin(dir_rad)
    return x_comp, y_comp


def sin_cos_scale(col, scale):
    """
    Scale a cyclical value e.g day of year, to a sin and cos function
    Args:
        col: value to be scaled
        scale: periodicity e.g for a hour timestamp 24, or 365 for a day of year

    Returns: a tuple of (sin_component, cos_component)

    """
    col = tf.cast(col, tf.float32)
    sin = tf.math.sin(col * (2 * math.pi / scale))
    cos = tf.math.cos(col * (2 * math.pi / scale))
    return sin, cos

def sin_cos_scale_np(col, scale):
    """
    Scale a cyclical value e.g day of year, to a sin and cos function
    Args:
        col: value to be scaled
        scale: periodicity e.g for a hour timestamp 24, or 365 for a day of year

    Returns: a tuple of (sin_component, cos_component)

    """
    # col = np.cast(col, np.float32)
    sin = np.sin(col * (2 * math.pi / scale))
    cos = np.cos(col * (2 * math.pi / scale))
    return sin, cos