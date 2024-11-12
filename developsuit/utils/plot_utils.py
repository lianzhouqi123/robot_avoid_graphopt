import numpy as np


def average_every_n(plot_list, n, keep_len=False):
    if len(plot_list) <= n:
        averages = plot_list
    else:
        if not keep_len:
            averages = [sum(plot_list[i:i + n]) / len(plot_list[i:i + n]) for i in range(0, len(plot_list), n)]
        else:
            if n % 2 == 0:
                n += 1
            averages_start = [sum(plot_list[0: int(i + (n + 1) / 2)]) / (i + (n + 1) / 2)
                              for i in range(int((n - 1) / 2))]

            averages_mid = [sum(plot_list[i - int((n - 1) / 2): i + int((n + 1) / 2)]) / n
                            for i in range(int((n - 1) / 2), len(plot_list) - int((n - 1) / 2))]

            averages_end = [sum(plot_list[i - int((n - 1) / 2):]) / len(plot_list[i - int((n - 1) / 2):])
                            for i in range(len(plot_list) - int((n - 1) / 2), len(plot_list))]

            averages = averages_start + averages_mid + averages_end

    return averages
