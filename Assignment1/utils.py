import math
import numpy as np

# Img standardize
def img_standardize(x):
    return np.divide(x, 255, dtype=float)

# standardize
def standardize(x):
    temp = x
    x -= x.mean(axis=0)
    x /= x.std(axis=0)

    X_std = temp
    mean = temp.mean(axis=0)
    std = temp.std(axis=0)
    for col in range(np.shape(temp)[1]):
        if std[col]:
            X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]

    return np.sum(x - X_std)
    #return x

def normalize(X, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)

def shuffle(x, y):
    idxs = np.arange(x.shape[0])
    np.random.shuffle(idxs)
    return x[idxs], y[idxs]

def one_hot_vectorize(categorical_y):
    y = np.zeros((len(categorical_y), 10))
    y[np.arange(0, len(categorical_y)), categorical_y] = 1
    return y

def preprocess(x_train, x_test, y_train, y_test, preprocess_type='normalize'):

    n_training_datapoints = x_train.shape[0]
    n_test_datapoints = x_test.shape[0]
    x_train = x_train.reshape(n_training_datapoints, -1)
    x_test = x_test.reshape(n_test_datapoints, -1)

    x_train = np.hstack((x_train, np.ones((n_training_datapoints, 1))))
    x_test = np.hstack((x_test, np.ones((n_test_datapoints, 1))))

    if preprocess_type == 'normalize':
        x_train = normalize(x_train)
        x_test = normalize(x_test)

    elif preprocess_type == 'standardize':
        x_train = standardize(x_train)
        x_test = standardize(x_test)

    elif preprocess_type == 'img_standardize':
        x_train = img_standardize(x_train)
        x_test = img_standardize(x_test)

    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

    y_train = one_hot_vectorize(y_train)

    return x_train, x_test, y_train, y_test