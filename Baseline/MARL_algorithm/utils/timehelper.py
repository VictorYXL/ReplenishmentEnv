import time

import numpy as np


def print_time(start_time, T, t_max, episode, episode_rewards):
    time_elapsed = time.time() - start_time
    T = max(1, T)
    time_left = time_elapsed * (t_max - T) / T
    # Just in case its over 100 days
    time_left = min(time_left, 60 * 60 * 24 * 100)
    last_reward = "N\A"
    if len(episode_rewards) > 5:
        last_reward = "{:.2f}".format(np.mean(episode_rewards[-50:]))
    print(
        "\033[F\033[F\x1b[KEp: {:,}, T: {:,}/{:,}, Reward: {}, \n\x1b[KElapsed: {}, Left: {}\n".format(
            episode, T, t_max, last_reward, time_str(time_elapsed), time_str(time_left)
        ),
        " " * 10,
        end="\r",
    )


def time_left(start_time, t_start, t_current, t_max):
    if t_current >= t_max:
        return "-"
    time_elapsed = time.time() - start_time
    t_current = max(1, t_current)
    time_left = time_elapsed * (t_max - t_current) / (t_current - t_start)
    # Just in case its over 100 days
    time_left = min(time_left, 60 * 60 * 24 * 100)
    return time_str(time_left)


def time_str(s):
    """
    Convert seconds to a nicer string showing days, hours, minutes and seconds
    """
    days, remainder = divmod(s, 60 * 60 * 24)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    string = ""
    if days > 0:
        string += "{:d} days, ".format(int(days))
    if hours > 0:
        string += "{:d} hours, ".format(int(hours))
    if minutes > 0:
        string += "{:d} minutes, ".format(int(minutes))
    string += "{:d} seconds".format(int(seconds))
    return string


class TimeStat(object):
    """A time stat for logging the elapsed time of code running
    Example:
        time_stat = TimeStat()
        with time_stat:
            // some code
        print(time_stat.mean)
    """

    def __init__(self, window_size=1):
        self.time_samples = WindowStat(window_size)
        self._start_time = None

    def __enter__(self):
        self._start_time = time.time()

    def __exit__(self, type, value, tb):
        time_delta = time.time() - self._start_time
        self.time_samples.add(time_delta)

    @property
    def mean(self):
        return self.time_samples.mean

    @property
    def min(self):
        return self.time_samples.min

    @property
    def max(self):
        return self.time_samples.max


class WindowStat(object):
    """Tool to maintain statistical data in a window."""

    def __init__(self, window_size):
        self.items = [None] * window_size
        self.idx = 0
        self.count = 0

    def add(self, obj):
        self.items[self.idx] = obj
        self.idx += 1
        self.count += 1
        self.idx %= len(self.items)

    @property
    def mean(self):
        if self.count > 0:
            return np.mean(self.items[: self.count])
        else:
            return None

    @property
    def min(self):
        if self.count > 0:
            return np.min(self.items[: self.count])
        else:
            return None

    @property
    def max(self):
        if self.count > 0:
            return np.max(self.items[: self.count])
        else:
            return None
