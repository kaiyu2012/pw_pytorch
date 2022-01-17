import matplotlib.pyplot as plt
import numpy as np

ratios = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]

#--------------level- 3-------------
#best val acc
top_k_l3_best_val_acc_reset = np.array([0.706, 0.716, 0.64, 0.642, 0.65, 0.542, 0.61, 0.558, 0.5680000000000001, 0.582, 0.5720000000000001, 0.582, 0.5660000000000001, 0.578, 0.586, 0.5740000000000001, 0.544, 0.5660000000000001, 0.5640000000000001, 0.578, 0.5640000000000001])

#--------------level- 3-------------
#best val acc
top_k_l3_best_val_acc_nonreset = np.array([0.764, 0.716, 0.662, 0.64, 0.61, 0.626, 0.63, 0.59, 0.5760000000000001, 0.578, 0.54, 0.582, 0.552, 0.534, 0.544, 0.552, 0.5760000000000001, 0.546, 0.556, 0.546, 0.548])

#--------------level- 3-------------
#best val acc
#m2_synflow_l3_best_val_acc = np.array([0.7400, 0.7060, 0.6800, 0.6360, 0.6080, 0.6120, 0.5360, 0.5800, 0.5560, 0.5940, 0.5780, 0.5940, 0.6180, 0.5920, 0.5980, 0.6000, 0.6020, 0.5940, 0.6180, 0.6080, 0.5940])


fig = plt.figure(figsize=(6, 5))


ax = fig.add_subplot(1, 1, 1)

ax.plot(ratios, top_k_l3_best_val_acc_reset, color= 'purple', ls='solid', linewidth = 0.5, marker = "^", label = "top_k_reset")
ax.plot(ratios, top_k_l3_best_val_acc_nonreset, color= 'blue', ls='solid', linewidth = 0.5, marker = "*", label = "top_k_nonreset")
#ax.plot(ratios, m2_synflow_l3_best_val_acc, color= 'red', ls='solid', linewidth = 0.5, marker = "x", label = "M2 SF (I=3)")

ax.set_facecolor("whitesmoke")
plt.grid(color = 'gray', linestyle = 'solid', linewidth = 0.5)

ax.set_title('Multi-shot SF Adjacency Pruning Comparisons (Cora)')
ax.set_xlabel('Compression Ratio')
ax.set_ylabel('Top-1 Accuracy')
ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
ax.set_xticks((0, 1, 2, 3, 4, 5))
plt.legend(loc=1)
plt.show()