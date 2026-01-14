# %%
import os

import matplotlib.pyplot as plt
import numpy as np


# generate a time series with a given length, there is a increasing trend. At certrain time, there is an abnormal spike impulse.
def generate_ts(length, spike_time, spike_value, trend_rate=0.1):
    ts = np.zeros(length)
    for i in range(length):
        ts[i] = ts[i - 1] + trend_rate

    seasonal_period = 10
    seasonal_amplitude = 0.5
    seasonal_phase = 0
    seasonal_frequency = 1 / seasonal_period
    seasonal_ts = seasonal_amplitude * np.sin(
        2 * np.pi * seasonal_frequency * np.arange(length) + seasonal_phase
    )
    ts += seasonal_ts

    ts[spike_time] = spike_value
    return ts


# ts = generate_ts(100, 50, 10)
ts = generate_ts(120, 50, 8, 0.05)
plt.plot(ts)
plt.show()
print(ts[:100], ts[100:])


# %%
# a step function
def generate_ts(length, jump_time, jump_value):
    ts = np.ones(length)
    for i in range(jump_time, length):
        ts[i] = jump_value

    seasonal_period = 10
    seasonal_amplitude = 0.2
    seasonal_phase = 0
    seasonal_frequency = 1 / seasonal_period
    seasonal_ts = seasonal_amplitude * np.sin(
        2 * np.pi * seasonal_frequency * np.arange(length) + seasonal_phase
    )
    ts += seasonal_ts

    return ts


# ts = generate_ts(100, 50, 10)
ts = generate_ts(120, 80, 8)
plt.plot(ts)
plt.show()
print(ts[:100], ts[100:])
# %%
