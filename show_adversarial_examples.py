"""
This script visualizes the adversarial samples on a grid.
How to run: python visualize_adv_examples.py ./location_of_bulk_pickle.pkl
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys, os

pkl_loc = './bulk_advgan2.pkl'

with open(pkl_loc, "rb") as f:
    adv_data_dict = pickle.load(f)

xs = adv_data_dict["xs"]
y_trues = adv_data_dict["y_trues"]
y_preds = adv_data_dict["y_preds"]
noises = adv_data_dict["noises"]
y_preds_adversarial = adv_data_dict["y_preds_adversarial"]

# visualize N random images
print('len xs', len(xs))
idxs = np.random.choice(range(500), size=(20,), replace=False)
for matidx, idx in enumerate(idxs):
    orig_im = xs[idx].reshape(28, 28)
    adv_im = orig_im + noises[idx].reshape(28, 28)
    disp_im = np.concatenate((orig_im, adv_im), axis=1)
    plt.subplot(5, 4, matidx + 1)
    plt.imshow(disp_im, "gray")  # plt.imshow()函数负责对图像进行处理，
    plt.xticks([])  # 并显示其格式，但是不能显示。其后跟着plt.show()才能显示出来。
    plt.yticks([])
    plt.title("Orig: {} | New: {}".format(y_trues[idx], y_preds_adversarial[idx]))
plt.show()

# Noise statistics
noises, xs, y_trues, y_preds = np.array(noises), np.array(xs), np.array(y_trues), np.array(y_preds)
adv_exs = xs + noises
print("Adv examples: max, min: ", adv_exs.max(), adv_exs.min())
print("Noise: Mean, Max, Min: ")
print(np.mean(noises), np.max(noises), np.min(noises))