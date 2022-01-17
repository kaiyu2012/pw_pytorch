from matplotlib import pyplot as plt
import numpy as np

def calculate(numpy_array):
    mean = numpy_array.mean(axis=0)
    minimum = np.amin(numpy_array,axis=0)
    maximum = np.amax(numpy_array,axis=0)
    return mean, minimum, maximum
ratios = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]

top_k_singleshot_run1 = np.array([0.758, 0.714, 0.674, 0.674, 0.662, 0.588, 0.548, 0.56, 0.6, 0.604, 0.616, 0.608, 0.586, 0.598, 0.588, 0.592, 0.592, 0.588, 0.588, 0.588, 0.588])
top_k_singleshot_run2 = np.array([0.762, 0.708, 0.694, 0.676, 0.626, 0.576, 0.576, 0.586, 0.582, 0.548, 0.538, 0.556, 0.544, 0.548, 0.548, 0.552, 0.552, 0.554, 0.554, 0.554, 0.554])
top_k_singleshot_run3 = np.array([0.764, 0.73, 0.68, 0.65, 0.646, 0.644, 0.592, 0.556, 0.562, 0.584, 0.602, 0.586, 0.598, 0.584, 0.564, 0.584, 0.584, 0.584, 0.584, 0.584, 0.584])

top_k_singleshot_mean, top_k_singleshot_min, top_k_singleshot_max = calculate(np.array([top_k_singleshot_run1,top_k_singleshot_run2,top_k_singleshot_run3]))

top_k_multishot_l3_run1 = np.array([0.77, 0.716, 0.682, 0.658, 0.64, 0.586, 0.604, 0.596, 0.594, 0.582, 0.578, 0.582, 0.59, 0.5740000000000001, 0.582, 0.5700000000000001, 0.594, 0.5760000000000001, 0.55, 0.582, 0.5760000000000001])
top_k_multishot_l3_run2 = np.array([0.794, 0.714, 0.686, 0.656, 0.662, 0.5760000000000001, 0.512, 0.5700000000000001, 0.56, 0.578, 0.592, 0.596, 0.606, 0.59, 0.594, 0.592, 0.588, 0.596, 0.594, 0.58, 0.588])
top_k_multishot_l3_run3 = np.array([0.8140000000000001, 0.724, 0.682, 0.646, 0.598, 0.538, 0.552, 0.56, 0.554, 0.5660000000000001, 0.59, 0.56, 0.5760000000000001, 0.552, 0.54, 0.582, 0.5680000000000001, 0.5740000000000001, 0.554, 0.5640000000000001, 0.5700000000000001])

top_k_multishot_mean, top_k_multishot_min, top_k_multishot_max = calculate(np.array([top_k_multishot_l3_run1,top_k_multishot_l3_run2,top_k_multishot_l3_run3]))

synflow_singleshot_run1 = np.array([0.774, 0.724, 0.676, 0.654, 0.614, 0.578, 0.52, 0.532, 0.56, 0.59, 0.594, 0.598, 0.608, 0.61, 0.61, 0.604, 0.604, 0.606, 0.606, 0.606, 0.606])
synflow_singleshot_run2 = np.array([0.766, 0.73, 0.694, 0.69, 0.676, 0.628, 0.602, 0.516, 0.57, 0.576, 0.574, 0.59, 0.598, 0.596, 0.604, 0.578, 0.578, 0.58, 0.58, 0.58, 0.58])
synflow_singleshot_run3 = np.array([0.792, 0.728, 0.704, 0.678, 0.65, 0.632, 0.596, 0.606, 0.584, 0.622, 0.632, 0.624, 0.62, 0.62, 0.608, 0.606, 0.606, 0.604, 0.604, 0.604, 0.604])

synflow_singleshot_mean, synflow_singleshot_min, synflow_singleshot_max = calculate(np.array([synflow_singleshot_run1,synflow_singleshot_run2,synflow_singleshot_run3]))

synflow_multishot_l3_run1 = np.array([0.742, 0.732, 0.686, 0.68, 0.6, 0.604, 0.604, 0.56, 0.594, 0.608, 0.61, 0.598, 0.592, 0.592, 0.6, 0.606, 0.59, 0.596, 0.592, 0.596, 0.61])
synflow_multishot_l3_run2 = np.array([0.778, 0.714, 0.672, 0.674, 0.648, 0.5680000000000001, 0.5700000000000001, 0.5720000000000001, 0.5660000000000001, 0.56, 0.604, 0.596, 0.588, 0.598, 0.596, 0.58, 0.602, 0.616, 0.608, 0.584, 0.604])
synflow_multishot_l3_run3 = np.array([0.754, 0.73, 0.672, 0.616, 0.63, 0.5640000000000001, 0.544, 0.544, 0.578, 0.554, 0.556, 0.5720000000000001, 0.588, 0.5700000000000001, 0.592, 0.584, 0.578, 0.578, 0.5700000000000001, 0.5640000000000001, 0.5760000000000001])

synflow_multishot_mean, synflow_multishot_min, synflow_multishot_max = calculate(np.array([synflow_multishot_l3_run1,synflow_multishot_l3_run2,synflow_multishot_l3_run3]))

ugs_run1 = np.array([0.792, 0.668, 0.574, 0.568, 0.542, 0.526, 0.542, 0.576, 0.59, 0.602, 0.592, 0.604, 0.588, 0.592, 0.596, 0.598, 0.598, 0.586, 0.586, 0.586, 0.586])
ugs_run2 = np.array([0.782, 0.732, 0.676, 0.652, 0.65, 0.634, 0.622, 0.618, 0.618, 0.61, 0.58, 0.558, 0.56, 0.556, 0.556, 0.568, 0.568, 0.554, 0.554, 0.554, 0.554])
ugs_run3 = np.array([0.786, 0.704, 0.646, 0.618, 0.61, 0.598, 0.594, 0.594, 0.594, 0.576, 0.584, 0.584, 0.576, 0.57, 0.574, 0.552, 0.552, 0.55, 0.55, 0.55, 0.55])
ugs_mean, ugs_min, ugs_max = calculate(np.array([ugs_run1, ugs_run2, ugs_run3]))

random_run1 = np.array([0.788, 0.68, 0.612, 0.558, 0.592, 0.58, 0.594, 0.558, 0.596, 0.602, 0.604, 0.594, 0.588, 0.61, 0.606, 0.606, 0.604, 0.606, 0.606, 0.606, 0.606])
random_run2 = np.array([0.792, 0.72, 0.66, 0.54, 0.528, 0.546, 0.548, 0.572, 0.566, 0.582, 0.588, 0.588, 0.592, 0.574, 0.592, 0.592, 0.59, 0.592, 0.592, 0.592, 0.592])
random_run3 = np.array([0.798, 0.658, 0.6, 0.544, 0.528, 0.504, 0.528, 0.536, 0.566, 0.532, 0.546, 0.556, 0.552, 0.556, 0.548, 0.554, 0.554, 0.554, 0.556, 0.552, 0.554])
random_mean, random_min, random_max = calculate(np.array([random_run1, random_run2, random_run3]))

admm_run1 = np.array([0.75646, 0.73063, 0.68635, 0.64576, 0.61255, 0.59779, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941])
admm_run2 = np.array([0.73063, 0.72694, 0.6679, 0.60517, 0.59779, 0.59779, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941])
admm_run3 = np.array([0.7048, 0.67159, 0.64576, 0.63469, 0.59041, 0.5941, 0.59779, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941, 0.5941])
admm_mean, admm_min, admm_max = calculate(np.array([admm_run1, admm_run2, admm_run3]))

plt.rc('font', family='serif', size=12)
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')

fig = plt.figure(figsize=(6, 6))

ax = fig.add_subplot(1, 1, 1)

ax.plot(ratios, top_k_singleshot_mean, color='magenta', ls='solid', linewidth = 2.0, marker = "x", label = "Top-k (Single Shot)")
ax.plot(ratios, top_k_multishot_mean, color='purple', ls='solid', linewidth = 2.0, marker = "*", label = "Top-k (Multiple Shot)")
ax.plot(ratios, synflow_singleshot_mean, color='orange', ls='solid', linewidth = 0.5, marker = "v", label = "SF (Single Shot)")
ax.plot(ratios, synflow_multishot_mean, color='blue', ls='solid', linewidth = 0.5, marker = "^", label = "SF (Multiple Shot)")

ax.plot(ratios, ugs_mean, color='red', ls='solid', linewidth = 0.5, marker = "s", label = "UGS")
#ax.fill_between(ratios, ugs_min, ugs_max, facecolor='lightcoral', alpha=0.2)

ax.plot(ratios, random_mean, color='green', ls='solid', linewidth = 0.5, marker = "o", label = "Random")
#ax.fill_between(ratios, random_min, random_max, facecolor='lightgreen', alpha=0.2)

ax.plot(ratios, admm_mean, color='mediumturquoise', ls='solid', linewidth = 1.0, marker = ">", label = "ADMM")

ax.set_facecolor("whitesmoke")
plt.grid(color = 'gray', linestyle = 'solid', linewidth = 0.5)

title = 'Cora_Adjacency_Pruning'
ax.set_title(title)
ax.set_xlabel('Compression Ratio')
ax.set_ylabel('Top-1 Accuracy')
ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
ax.set_xticks((0, 1, 2, 3, 4, 5))
plt.legend(loc = 4, prop={'size':12})

plt.savefig('figure.png')
plt.show()
