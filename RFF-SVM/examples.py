import numpy as np
from sklearn.svm import SVC
import time
def RBF_kernel_fn(x1, x2):
    """The RBF kernel

    Args:
        x1 (np.ndarray): x1.shape=(x1_num, d)
        x2 (np.ndarray): x2.shape=(x2_num, d)

    Returns:
        gram (np.ndarray): gram.shape=(x1_num, x2_num)
    """
    x1 = np.expand_dims(x1, 1)
    x2 = np.expand_dims(x2, 0)
    delta = (x1 - x2)
    gram = np.exp(-np.sum(delta**2, axis=-1)/2)
    return gram

def linear_kernel_fn(x1, x2):
    """The linear kernel

    Args:
        x1 (np.ndarray): x1.shape=(x1_num, d)
        x2 (np.ndarray): x2.shape=(x2_num, d)

    Returns:
        gram (np.ndarray): gram.shape=(x1_num, x2_num)
    """
    return np.dot(x1, x2.T)

def train_kernel_svm_classifier_with_gram(x, y, kernel_fn):
    clf = SVC(kernel="precomputed")
    gram_train = kernel_fn(x, x)
    clf.fit(gram_train, y)
    start = time.perf_counter()
    acc = clf.score(gram_train, y)
    end = time.perf_counter()
    print(end - start)
    return clf