# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import numpy as np

# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    a = torch.tensor([i for i in range(10)])
    b = torch.tensor([i for i in range(10)])
    c = [i for i in range(10)]
    d = [i for i in range(10)]
    f = [i for i in range(10)]
    e = np.zeros((10, 3))
    e[:, 0] = c
    e[:, 1] = d
    e[:, 2] = f
    g = e.reshape(-1, 5, 2)
    print(e[:, [0, 1]])


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
