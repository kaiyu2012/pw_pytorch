from matplotlib import pyplot as plt
import numpy as np

def calculate(numpy_array):
    mean = numpy_array.mean(axis=0)
    minimum = np.amin(numpy_array,axis=0)
    maximum = np.amax(numpy_array,axis=0)
    return mean, minimum, maximum
ratios = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

top_k_singleshot_run1 = np.array([0.618, 0.656, 0.656, 0.624, 0.592, 0.568, 0.554, 0.568, 0.55, 0.532, 0.544, 0.534, 0.534, 0.526, 0.542, 0.534, 0.536, 0.536, 0.536, 0.536, 0.536])[0:9]
top_k_singleshot_run2 = np.array([0.692, 0.652, 0.654, 0.634, 0.596, 0.578, 0.598, 0.542, 0.566, 0.568, 0.526, 0.532, 0.538, 0.546, 0.53, 0.526, 0.512, 0.512, 0.512, 0.512, 0.512])[0:9]
top_k_singleshot_run3 = np.array([0.654, 0.632, 0.636, 0.61, 0.604, 0.558, 0.584, 0.564, 0.562, 0.544, 0.532, 0.536, 0.55, 0.554, 0.552, 0.558, 0.542, 0.542, 0.542, 0.542, 0.542])[0:9]

top_k_singleshot_mean, top_k_singleshot_min, top_k_singleshot_max = calculate(np.array([top_k_singleshot_run1,top_k_singleshot_run2,top_k_singleshot_run3]))

top_k_multishot_l3_run1 = np.array([0.674, 0.656, 0.648, 0.664, 0.608, 0.552, 0.542, 0.518, 0.504, 0.5, 0.49, 0.526, 0.536, 0.532, 0.552, 0.54, 0.522, 0.536, 0.496, 0.482, 0.548])[0:9]
top_k_multishot_l3_run2 = np.array([0.726, 0.648, 0.662, 0.656, 0.582, 0.552, 0.56, 0.582, 0.582, 0.59, 0.516, 0.54, 0.52, 0.512, 0.536, 0.534, 0.526, 0.506, 0.504, 0.52, 0.52])[0:9]
top_k_multishot_l3_run3 = np.array([0.658, 0.624, 0.652, 0.63, 0.644, 0.598, 0.536, 0.5700000000000001, 0.586, 0.584, 0.562, 0.542, 0.522, 0.496, 0.494, 0.49, 0.48, 0.498, 0.52, 0.494, 0.488])[0:9]

top_k_multishot_mean, top_k_multishot_min, top_k_multishot_max = calculate(np.array([top_k_multishot_l3_run1,top_k_multishot_l3_run2,top_k_multishot_l3_run3]))

synflow_singleshot_run1 = np.array([0.672, 0.666, 0.654, 0.626, 0.612, 0.584, 0.602, 0.586, 0.534, 0.536, 0.56, 0.556, 0.55, 0.548, 0.536, 0.536, 0.538, 0.538, 0.538, 0.538, 0.538])[0:9]
synflow_singleshot_run2 = np.array([0.672, 0.656, 0.662, 0.572, 0.584, 0.584, 0.582, 0.576, 0.548, 0.54, 0.54, 0.54, 0.54, 0.554, 0.554, 0.554, 0.546, 0.546, 0.546, 0.546, 0.546])[0:9]
synflow_singleshot_run3 = np.array([0.68, 0.652, 0.66, 0.642, 0.63, 0.608, 0.592, 0.574, 0.588, 0.536, 0.552, 0.562, 0.57, 0.572, 0.548, 0.55, 0.556, 0.556, 0.556, 0.556, 0.556])[0:9]

synflow_singleshot_mean, synflow_singleshot_min, synflow_singleshot_max = calculate(np.array([synflow_singleshot_run1,synflow_singleshot_run2,synflow_singleshot_run3]))

synflow_multishot_l3_run1 = np.array([0.686, 0.644, 0.662, 0.614, 0.62, 0.606, 0.604, 0.612, 0.584, 0.584, 0.538, 0.554, 0.516, 0.53, 0.512, 0.526, 0.504, 0.498, 0.534, 0.532, 0.536])[0:9]
synflow_multishot_l3_run2 = np.array([0.6960000000000001, 0.65, 0.672, 0.656, 0.652, 0.588, 0.614, 0.598, 0.582, 0.5700000000000001, 0.556, 0.5760000000000001, 0.59, 0.5700000000000001, 0.5680000000000001, 0.548, 0.582, 0.554, 0.5660000000000001, 0.5660000000000001, 0.5660000000000001])[0:9]
synflow_multishot_l3_run3 = np.array([0.682, 0.63, 0.62, 0.628, 0.622, 0.606, 0.582, 0.522, 0.552, 0.578, 0.582, 0.558, 0.534, 0.542, 0.54, 0.536, 0.518, 0.548, 0.518, 0.524, 0.52])[0:9]

synflow_multishot_mean, synflow_multishot_min, synflow_multishot_max = calculate(np.array([synflow_multishot_l3_run1,synflow_multishot_l3_run2,synflow_multishot_l3_run3]))

ugs_run1 = np.array([0.698, 0.566, 0.55, 0.534, 0.536, 0.548, 0.56, 0.554, 0.568, 0.554, 0.54, 0.54, 0.556, 0.558, 0.562, 0.556, 0.55, 0.55, 0.55, 0.55, 0.55])[0:9]
ugs_run2 = np.array([0.714, 0.534, 0.512, 0.482, 0.478, 0.468, 0.476, 0.482, 0.506, 0.526, 0.562, 0.554, 0.558, 0.562, 0.558, 0.558, 0.568, 0.568, 0.568, 0.568, 0.568])[0:9]
ugs_run3 = np.array([0.71, 0.552, 0.564, 0.544, 0.536, 0.526, 0.534, 0.532, 0.522, 0.548, 0.542, 0.554, 0.544, 0.538, 0.534, 0.538, 0.536, 0.536, 0.536, 0.536, 0.536])[0:9]
ugs_mean, ugs_min, ugs_max = calculate(np.array([ugs_run1, ugs_run2, ugs_run3]))

random_run1 = np.array([0.704, 0.618, 0.548, 0.534, 0.512, 0.54, 0.548, 0.552, 0.566, 0.53, 0.546, 0.538, 0.542, 0.542, 0.544, 0.544, 0.544, 0.544, 0.544, 0.544, 0.544])[0:9]
random_run2 = np.array([0.714, 0.564, 0.57, 0.504, 0.48, 0.526, 0.504, 0.486, 0.518, 0.492, 0.498, 0.496, 0.51, 0.49, 0.492, 0.492, 0.492, 0.492, 0.492, 0.492, 0.492])[0:9]
random_run3 = np.array([0.698, 0.652, 0.598, 0.508, 0.554, 0.536, 0.536, 0.566, 0.552, 0.544, 0.542, 0.562, 0.552, 0.552, 0.546, 0.552, 0.552, 0.552, 0.552, 0.552, 0.544])[0:9]
random_mean, random_min, random_max = calculate(np.array([random_run1, random_run2, random_run3]))


admm_run1 = np.array([0.652, 0.64, 0.608, 0.594, 0.576, 0.577, 0.577, 0.577, 0.577, 0.577, 0.577, 0.577, 0.577, 0.577, 0.577, 0.577, 0.577, 0.577, 0.577, 0.577, 0.577])[0:9]
admm_run2 = np.array([0.614, 0.611, 0.59, 0.562, 0.558, 0.559, 0.559, 0.559, 0.559, 0.559, 0.559, 0.559, 0.559, 0.559, 0.559, 0.559, 0.559, 0.559, 0.559, 0.559, 0.559])[0:9]
admm_run3 = np.array([0.616, 0.602, 0.568, 0.565, 0.547, 0.548, 0.548, 0.548, 0.548, 0.548, 0.548, 0.548, 0.548, 0.548, 0.548, 0.548, 0.548, 0.548, 0.548, 0.548, 0.548])[0:9]
admm_mean, admm_min, admm_max = calculate(np.array([admm_run1, admm_run2, admm_run3]))

plt.rc('font', family='serif', size=12)
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')

fig = plt.figure(figsize=(6, 6))

ax = fig.add_subplot(1, 1, 1)
ax.plot(ratios, top_k_multishot_mean, color='purple', ls='solid', linewidth = 0.5, marker = "*", label = "IGRP (Multiple Shot)")
ax.plot(ratios, top_k_singleshot_mean, color='magenta', ls='solid', linewidth = 0.5, marker = "x", label = "IGRP (Single Shot)")
ax.plot(ratios, synflow_multishot_mean, color='blue', ls='solid', linewidth = 0.5, marker = "^", label = "SF (Multiple Shot)")
ax.plot(ratios, synflow_singleshot_mean, color='orange', ls='solid', linewidth = 0.5, marker = "v", label = "SF (Single Shot)")


ax.plot(ratios, ugs_mean, color='red', ls='solid', linewidth = 0.5, marker = "s", label = "UGS")
#ax.fill_between(ratios, ugs_min, ugs_max, facecolor='lightcoral', alpha=0.2)


#ax.fill_between(ratios, random_min, random_max, facecolor='lightgreen', alpha=0.2)
ax.plot(ratios, admm_mean, color='mediumturquoise', ls='solid', linewidth = 0.5, marker = ">", label = "ADMM")
ax.plot(ratios, random_mean, color='green', ls='solid', linewidth = 0.5, marker = "o", label = "Random")
ax.set_facecolor("whitesmoke")
plt.grid(color = 'gray', linestyle = 'solid', linewidth = 0.5)

title = 'Adjacency Pruning (Citeseer)'
ax.set_title(title)
ax.set_xlabel('Compression Ratio')
ax.set_ylabel('Top-1 Accuracy')
ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
ax.set_xticks((0, 0.5, 1, 1.5, 2))
plt.legend(loc = 4, prop={'size':12})
plt.savefig('figure.png')
plt.show()
