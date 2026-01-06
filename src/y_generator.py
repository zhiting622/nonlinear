import matplotlib as plt
import numpy as np

# The function that used to calculate for each single hr from the window
# N = window size
# s = shift size
def get_yi(x, i, N=8, s=2):
    # N in sec: between i*2 and 8+i*2
    # below is the N in idx
    start = i*s
    end = (N+i*s)

    x_sum = np.cumsum(x)

    # find the peaks
    peaks = []
    for times in x_sum:
        if times > start and times <= end:
            peaks.append(times)
        if times > end:
            break

    # Peak to peak intervals
    intervals = np.subtract(peaks[1:], peaks[:-1])

    # calculate the hr from P2P intervals
    yi = np.mean(intervals)

    return yi


def generate_y(x, theta, s=2):
    y = []
    total_length = int((np.cumsum(x)[-1]-theta)/s)
    for i in range(total_length):
        yi = get_yi(x, i, N=theta)
        y.append(yi)
    return y

