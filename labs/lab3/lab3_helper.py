# DO NOT EDIT!!!

import numpy as np
import matplotlib.pyplot as plt
import time

def plot_runtimes_2(input_sizes, avg_times, plot_title, ymax=None):
    plt.figure(figsize=(16, 10))
    plt.title(plot_title)
    plt.plot(input_sizes, avg_times)
    plt.ylabel("Average Runtime (seconds)")
    plt.xlabel("Input Size")
    plt.xlim([1, input_sizes[-1]])
    if ymax:
        plt.ylim([0, ymax])
    plt.show()

def plot_runtimes(input_sizes, dft_avg_times, my_avg_times, numpy_avg_times, plot_title, ymax=None):
    plt.figure(figsize=(16, 4))
    plt.title(plot_title)
    plt.plot(input_sizes, dft_avg_times)
    plt.plot(input_sizes, my_avg_times)
    plt.plot(input_sizes, numpy_avg_times)
    plt.ylabel("Average Runtime (seconds)")
    plt.xlabel("Input Size")
    plt.legend(("Naive DFT", "Our FFT", "Numpy's FFT"))
    plt.xlim([1, input_sizes[-1]])
    if ymax:
        plt.ylim([0, ymax])
    plt.show()

def get_avg_runtimes(f, input_sizes, num_trials, zero_pad=False):
    """
    Given an FFT function f and an iterable type (list, range, etc.) of input_sizes,
    returns the average (over num_trials) trials runtime of f for randomly generated 
    data of each size within input_sizes.
    """
    runtimes = []
    for n in input_sizes:
        data = random_complex_array(n)
        if zero_pad:
            L = next_power_of(2, len(data)) - len(data)
            data = np.concatenate((data, np.zeros(L)))
        runtimes.append(time_execution(f, data, num_trials))
    return runtimes

def next_power_of(k, n):
    """
    Returns the next power of k after and including the number n.
    Not numerically stable.
    
    >>> next_power_of(2, 5)     # next power of 2 after/including 5
    8
    >>> next_power_of(2, 16)    # next power of 2 after/including 16
    16
    >>> next_power_of(3, 81)    # next power of 3 after/including 81
    81
    >>> next_power_of(3, 28)    # next power of 3 after/including 28
    81
    """
    res = np.ceil(np.log(n) / np.log(k))
    return k ** res.astype('int')

def time_execution(f, arg, num_trials):
    """
    Returns the runtime of a single argument function f when called on arg,
    averaged over num_trials trials.
    """
    times = []
    for _ in range(num_trials):
        t0 = time.time()
        f(arg)
        tf = time.time()
        times.append(tf - t0)
    return sum(times) / len(times)

def random_complex_array(N):
    """
    Generates a length N numpy array of complex numbers whose real and imaginary 
    parts are both chosen uniformly at random from the interval [0, 1).
    """
    re = np.random.rand(N)
    im = np.random.rand(N)
    return re + 1j*im

def run_fft_tests(my_fft):
    """
    Run tests comparing FFT functions f, g, and display information about
    pass/fail and runtime. This is not meant to be used for performance
    profiling, rather just a ballpark for whether or not the function's
    runtimes are reasonable, as the time will also include printing time
    which technically shouldn't be factored in.
    """
    ref_fft = np.fft.fft
    t0 = time.time()
    
    # three tests on fixed data
    x1 = np.array([1, 2, 3])
    print("Test 1 passed: {0}".format(np.allclose(my_fft(x1), ref_fft(x1))))

    x2 = np.array([-1, 1j, -1, 1j])
    print("Test 2 passed: {0}".format(np.allclose(my_fft(x2), ref_fft(x2))))

    x3 = np.zeros(2 ** 8)
    print("Test 3 passed: {0}".format(np.allclose(my_fft(x3), ref_fft(x3))))

    # randomized larger datasets to test on
    x4 = np.random.rand(2 ** 12)
    print("Test 4 passed: {0}".format(np.allclose(my_fft(x4), ref_fft(x4))))

    x5 = random_complex_array(2 ** 12)
    print("Test 5 passed: {0}".format(np.allclose(my_fft(x5), ref_fft(x5))))
    
    tf = time.time()
    print("Tests took {0} seconds".format(round(tf - t0, 3)))