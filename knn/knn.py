#encoding=utf-8
from cs231n.data_utils import load_CIFAR10

import numpy as np
from cs231n.classifiers import KNearestNeighbor

def main():
    X_train, y_train, X_test, y_test = load_CIFAR10('../cifar-10-batches-py')

    num_training = 48000
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]

    num_test = 1000
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Reshape the image data into rows
    print(X_train.shape)
    '''
    (48000, 32, 32, 3)
    '''
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    print(X_train.shape)
    '''
    (48000, 3072)
    '''
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    print(X_train.shape, X_test.shape)
    '''
    (48000, 3072) (1000, 3072)
    '''
    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)
    y_test_pred = classifier.predict(X_test, k=5)
    print(y_test_pred)

    # Compute and display the accuracy
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
    '''
    Got 348 / 1000 correct => accuracy: 0.348000
    '''

if __name__ == "__main__":
    main()