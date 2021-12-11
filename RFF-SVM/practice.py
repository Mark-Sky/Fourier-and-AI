from pathlib import Path

import numpy as np
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt

from data import get_moon_data
from examples import RBF_kernel_fn, linear_kernel_fn, train_kernel_svm_classifier_with_gram
from utils import plot_decision_boundary, plot_data, plot_decision_boundary_sklearn
import time
Path("./results").mkdir(parents=True, exist_ok=True)

data_name = "moon_data"
data_x, data_y = get_moon_data()
r_features = 1000

def demo_plot_data():
    """
    plot distribution of datasets
    """
    plot_data(data_x, data_y, title=data_name,
              path="./results/{}.jpg".format(data_name))


def demo_linear_svm():
    clf = train_kernel_svm_classifier_with_gram(
        data_x, data_y, kernel_fn=linear_kernel_fn)
    plot_decision_boundary(data_x, data_y, clf, kernel_fn=linear_kernel_fn, boundary=True,
                           title="linear kernel prediction",
                           path="./results/linear_{}.jpg".format(data_name))


def demo_rbf_svm():
    print(data_x.shape)
    start = time.perf_counter()
    clf = train_kernel_svm_classifier_with_gram(
        data_x, data_y, kernel_fn=RBF_kernel_fn)
    end = time.perf_counter()
    print("time_cost = ", end - start)


    plot_decision_boundary(data_x, data_y, clf, kernel_fn=RBF_kernel_fn, boundary=True,
                           title="rbf kernel prediction",
                           path="./results/rbf_{}.jpg".format(data_name))



def RFF_kernel_fn(x1, x2):
    """
    Approximate RBF kernel with random fourier features.
    Reference:
        Random Features for Large-Scale Kernel Machines
        https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html

    Input: [n_sample, n_dim]
        x1.shape=(x1_n, d)
        x2.shape=(x2_n, d)
    Return:
        gram.shape=(x1_n, x2_n)

    TODO: Complete this function.
    """
    n_dims = x1.shape[1]
    w = np.random.normal(size=(n_dims, r_features))
    b = np.random.uniform(low=0.0, high=2*np.pi, size=r_features)
    zx_1 = np.sqrt(2 / r_features) * np.cos(x1 @ w + b)
    zx_2 = np.sqrt(2 / r_features) * np.cos(x2 @ w + b)
    gram = np.dot(zx_1, zx_2.T)
    return gram


def test_RFF_kernel_fn(x_dim):
    """
    TODO:
        1. investigate how the dimension of random fourier features affect the precision of approximation.
        2. investigate how x_dim affect the speed of rbf kernel and rff kernel.

    Reference:
        On the Error of Random Fourier Features, UAI 2015
        https://arxiv.org/abs/1506.02785
    """

    x1 = np.random.randn(x_dim, 2)
    x2 = np.random.randn(x_dim, 2)
    start_bf = time.perf_counter()
    gram_rbf = RBF_kernel_fn(x1, x2)
    end_bf = time.perf_counter()

    start_rf = time.perf_counter()
    gram_rff = RFF_kernel_fn(x1, x2)
    end_rf = time.perf_counter()
    # diff = np.max(np.abs(gram_rbf - gram_rff))
    diff = np.mean((gram_rbf - gram_rff)**2)
    print("MSE of gram matrix: {:.10f}".format(diff))
    return diff, (end_bf-start_bf), (end_rf - start_rf)
    # D=100000, MSE ≈ 1e-5


def test_RFF_kernel_svm():
    """Test how your RFF perform.
    """
    clf = train_kernel_svm_classifier_with_gram(
        data_x, data_y, kernel_fn=RFF_kernel_fn)

    plot_decision_boundary(data_x, data_y, clf, kernel_fn=RFF_kernel_fn, boundary=True,
                           title="D=10000 rff kernel prediction",
                           path="./results/10000rff_{}.jpg".format(data_name))

def RFF_approxiamte():

    n_dims = data_x.shape[1]
    w = np.random.normal(size=(n_dims, r_features))
    b = np.random.uniform(low=0.0, high=2 * np.pi, size=r_features)
    X = np.sqrt(2 / r_features) * np.cos(data_x @ w + b)
    print(data_x.shape)
    print(X.shape)



    '''
    plot_decision_boundary_sklearn(data_x, data_y, w, b, r_features, clf, boundary=True,
                                   title='Fast approximation of RBF-SVM Using RFF',
                                   path= "./results/FastRFF{}.jpg".format(data_name))
    '''
    start = time.perf_counter()
    clf = LinearSVC(random_state=0, tol=1e-5)
    clf.fit(X, data_y)
    #acc = clf.score(X, data_y)
    end = time.perf_counter()
    #print(acc)
    print("time_cost = ", end - start)

if __name__ == "__main__":
    '''
    demo_plot_data()
    demo_linear_svm()
    demo_rbf_svm()
    '''

    '''
    diff_list = []
    for i in range(10, 1000):
        r_features = i
        diff = test_RFF_kernel_fn()
        diff_list.append(diff)
    plt.plot(list(range(10,1000)), diff_list)
    plt.xlabel('D')
    plt.ylabel('diff')
    plt.show()
    '''

    '''
    bt_list, rt_list = [], []
    for i in range(100, 1000):
        diff, bt, rt = test_RFF_kernel_fn(i)
        bt_list.append(bt)
        rt_list.append(rt)
    plt.plot(list(range(100, 1000)), bt_list, c='red')
    plt.plot(list(range(100, 1000)), rt_list, c='blue')
    plt.legend(['RBF','RFF'])
    plt.xlabel('x_dim')
    plt.ylabel('time cost')
    plt.show()
    '''

    '''
    bt_list, rt_list = [], []
    for i in range(100, 10000):
        r_features = i
        diff, bt, rt = test_RFF_kernel_fn(100)
        bt_list.append(bt)
        rt_list.append(rt)
    #plt.plot(list(range(100, 1000)), bt_list, c='red')
    plt.plot(list(range(100, 10000)), rt_list, c='blue')
    #plt.legend(['RBF', 'RFF'])
    plt.xlabel('D')
    plt.ylabel('time cost')
    plt.show()
    '''
    #test_RFF_kernel_svm()
    #RFF_approxiamte()
    #demo_rbf_svm()
    demo_linear_svm()