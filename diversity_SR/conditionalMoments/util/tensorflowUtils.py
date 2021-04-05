import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
      value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(index,h_LR,w_LR,h_HR,w_HR,c,data_LR,data_HR):
    """
    Creates a tf.Example message ready to be written to a file.
    For writing files with LR and HR
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.

    feature = {
            'index'    : _int64_feature(index),
            'data_LR'   : _bytes_feature(data_LR),
            'h_LR'     : _int64_feature(h_LR),
            'w_LR'     : _int64_feature(w_LR),
            'data_HR'   : _bytes_feature(data_HR),
            'h_HR'     : _int64_feature(h_HR),
            'w_HR'     : _int64_feature(w_HR),
            'c'        : _int64_feature(c),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def diversity_image_example(index,data_LR,h_LR,w_LR,data_HR,h_HR,w_HR,c,mean,std):
    """
    Creates a tf.Example message ready to be written to a file.
    For writing files with LR, HR and conditional moments
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.

    feature = {
            'index'    : _int64_feature(index),
            'data_LR'  : _bytes_feature(data_LR),
            'h_LR'     : _int64_feature(h_LR),
            'w_LR'     : _int64_feature(w_LR),
            'data_HR'  : _bytes_feature(data_HR),
            'h_HR'     : _int64_feature(h_HR),
            'w_HR'     : _int64_feature(w_HR),
            'c'        : _int64_feature(c),
            'mean'     : _bytes_feature(mean),
            'std'      : _bytes_feature(std),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

def SF_image_example(index,h_LR,w_LR,h_SF,w_SF,c,data_LR,data_SF):
    """
    Creates a tf.Example message ready to be written to a file.
    For writing files with LR and SF
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.

    feature = {
            'index'    : _int64_feature(index),
            'data_LR'   : _bytes_feature(data_LR),
            'h_LR'     : _int64_feature(h_LR),
            'w_LR'     : _int64_feature(w_LR),
            'data_SF'   : _bytes_feature(data_SF),
            'h_SF'     : _int64_feature(h_SF),
            'w_SF'     : _int64_feature(w_SF),
            'c'        : _int64_feature(c),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

# Create a dictionary describing the features.
def _parse_image_wind_function(example_proto):
    """
    Dictionary for reading LR and HR
    """
    image_feature_description = {
        'index': tf.io.FixedLenFeature([], tf.int64),
        'data_LR': tf.io.FixedLenFeature([], tf.string),
        'h_LR': tf.io.FixedLenFeature([], tf.int64),
        'w_LR': tf.io.FixedLenFeature([], tf.int64),
        'data_HR': tf.io.FixedLenFeature([], tf.string),
        'h_HR': tf.io.FixedLenFeature([], tf.int64),
        'w_HR': tf.io.FixedLenFeature([], tf.int64),
        'c': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, image_feature_description)
    h_HR, w_HR, c = example['h_HR'], example['w_HR'], example['c']
    h_LR, w_LR, c = example['h_LR'], example['w_LR'], example['c']
    data_HR = tf.io.decode_raw(example['data_HR'], tf.float64)
    data_HR = tf.reshape(data_HR, (1 , h_HR, w_HR, c))
    data_LR = tf.io.decode_raw(example['data_LR'], tf.float64)
    data_LR = tf.reshape(data_LR, (1 , h_LR, w_LR, c))
    return data_HR, data_LR

def _parse_image_wind_diversityFull(example_proto):
    """
    Dictionary for reading LR and HR, and conditional moments
    """
    image_feature_description = {
        'index': tf.io.FixedLenFeature([], tf.int64),
        'data_LR': tf.io.FixedLenFeature([], tf.string),
        'h_LR': tf.io.FixedLenFeature([], tf.int64),
        'w_LR': tf.io.FixedLenFeature([], tf.int64),
        'data_HR': tf.io.FixedLenFeature([], tf.string),
        'h_HR': tf.io.FixedLenFeature([], tf.int64),
        'w_HR': tf.io.FixedLenFeature([], tf.int64),
        'c': tf.io.FixedLenFeature([], tf.int64),
        'mean': tf.io.FixedLenFeature([], tf.string),
        'std': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example_proto, image_feature_description)
    h_HR, w_HR, c = example['h_HR'], example['w_HR'], example['c']
    h_LR, w_LR, c = example['h_LR'], example['w_LR'], example['c']
    mean = tf.io.decode_raw(example['mean'], tf.float64)
    mean = tf.reshape(mean, (h_HR, w_HR, c))
    std = tf.io.decode_raw(example['std'], tf.float64)
    std = tf.reshape(std, (h_HR, w_HR, c))
    data_HR = tf.io.decode_raw(example['data_HR'], tf.float64)
    data_HR = tf.reshape(data_HR, (1 , h_HR, w_HR, c))
    data_LR = tf.io.decode_raw(example['data_LR'], tf.float64)
    data_LR = tf.reshape(data_LR, (1 , h_LR, w_LR, c))
    return data_HR, data_LR, mean, std

def _parse_image_wind_SFfunction(example_proto):
    """
    Dictionary for reading LR and SF
    """
    image_feature_description = {
        'index': tf.io.FixedLenFeature([], tf.int64),
        'data_LR': tf.io.FixedLenFeature([], tf.string),
        'h_LR': tf.io.FixedLenFeature([], tf.int64),
        'w_LR': tf.io.FixedLenFeature([], tf.int64),
        'data_SF': tf.io.FixedLenFeature([], tf.string),
        'h_SF': tf.io.FixedLenFeature([], tf.int64),
        'w_SF': tf.io.FixedLenFeature([], tf.int64),
        'c': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, image_feature_description)
    h_SF, w_SF, c = example['h_SF'], example['w_SF'], example['c']
    h_LR, w_LR, c = example['h_LR'], example['w_LR'], example['c']
    data_SF = tf.io.decode_raw(example['data_SF'], tf.float64)
    data_SF = tf.reshape(data_SF, (h_SF, w_SF, c))
    data_LR = tf.io.decode_raw(example['data_LR'], tf.float64)
    data_LR = tf.reshape(data_LR, (h_LR, w_LR, c))
    return data_LR, data_SF

