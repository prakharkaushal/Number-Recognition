import numpy as np

def load_data(path='mnist.npz'):
    """
    Loads the MNIST dataset from a .npz file.

    Args:
        path (str): The path to the .npz file.

    Returns:
        tuple: A tuple containing the training and testing data:
               (x_train, y_train), (x_test, y_test).
    """
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)
