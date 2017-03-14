import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def show_one(self):
    fig = self.get_figure()
    fig.savefig("/tmp/chart.png", dpi=90)
    fig.clear()

matplotlib.artist.Artist.show_one = show_one


# def problem_five_zeros(x):
#     if x > 2 or x <= 0:
#         return 0
#     return x/2

# def problem_five_ones(x):
#     if x > 4 or x <= 1:
#         return 0
#     return (x-1)/3


# sns.set_style("darkgrid")

# x0 = np.arrange(-1, 5, .01)
# plt.plot(x0, [problem_five_zeros(x) for x in x0])
# plt.show()


# plt.show_one()
