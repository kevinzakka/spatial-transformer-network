import numpy as np

from PIL import Image


def img2array(data_path, desired_size=None, expand=False, view=False):
    """Loads an RGB image as a 3D or 4D numpy array."""
    img = Image.open(data_path)
    img = img.convert('RGB')
    if desired_size:
        img = img.resize((desired_size[1], desired_size[0]))
    if view:
        img.show()
    x = np.asarray(img, dtype='float32')
    if expand:
        x = np.expand_dims(x, axis=0)
    x /= 255.0
    return x


def array2img(x):
    """Converts a numpy array to a PIL img."""
    x = np.asarray(x)
    x = x + max(-np.min(x), 0)
    x_max = np.max(x)
    if x_max != 0:
        x /= x_max
    x *= 255
    return Image.fromarray(x.astype('uint8'), 'RGB')


def deg2rad(x):
    """Converts an angle in degrees to radians."""
    return (x * np.pi) / 180
