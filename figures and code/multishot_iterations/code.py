import matplotlib.pyplot as plt
import numpy as np


def calculate(array_list):
    min_list = []
    max_list = []
    avg_list = []
    # 1000
    for i in range(len(array_list[0])):

        # 5
        list = []
        for j in range(len(array_list)):
            list.append(array_list[j][i])

        minimum = min(list)
        maximum = max(list)
        average = sum(list) / len(list)

        min_list.append(minimum)
        max_list.append(maximum)
        avg_list.append(average)
    return avg_list, min_list, max_list


ratios = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]

level_list = [1, 2, 3, 4, 5]

# Experiment variables
prune_strategy = ['synflow']

data = {'number_of_runs': 3, 'run_0': {'synflow': [{'level_1': [0.594, 0.638, 0.638, 0.64, 0.634, 0.63, 0.62, 0.534, 0.534, 0.49, 0.42, 0.378, 0.258, 0.258, 0.258, 0.258, 0.258, 0.258, 0.258, 0.258, 0.258], 'level_2': [0.594, 0.646, 0.642, 0.632, 0.63, 0.628, 0.63, 0.61, 0.602, 0.5740000000000001, 0.5, 0.462, 0.26, 0.246, 0.258, 0.258, 0.258, 0.258, 0.258, 0.258, 0.258], 'level_3': [0.592, 0.646, 0.638, 0.644, 0.644, 0.656, 0.624, 0.604, 0.586, 0.5720000000000001, 0.56, 0.372, 0.32, 0.256, 0.258, 0.234, 0.258, 0.258, 0.258, 0.268, 0.258], 'level_4': [0.598, 0.64, 0.63, 0.64, 0.62, 0.656, 0.66, 0.628, 0.598, 0.58, 0.508, 0.366, 0.40800000000000003, 0.332, 0.356, 0.294, 0.256, 0.258, 0.258, 0.258, 0.258], 'level_5': [0.584, 0.658, 0.658, 0.66, 0.636, 0.65, 0.658, 0.63, 0.614, 0.5640000000000001, 0.428, 0.522, 0.434, 0.366, 0.28200000000000003, 0.232, 0.254, 0.28, 0.258, 0.294, 0.258]}]}, 'run_1': {'synflow': [{'level_1': [0.608, 0.628, 0.63, 0.62, 0.604, 0.624, 0.61, 0.622, 0.616, 0.6, 0.488, 0.386, 0.214, 0.25, 0.25, 0.24, 0.202, 0.202, 0.232, 0.218, 0.24], 'level_2': [0.612, 0.634, 0.638, 0.628, 0.612, 0.626, 0.632, 0.606, 0.61, 0.536, 0.508, 0.47400000000000003, 0.358, 0.256, 0.216, 0.256, 0.24, 0.25, 0.218, 0.24, 0.212], 'level_3': [0.606, 0.66, 0.624, 0.636, 0.634, 0.64, 0.602, 0.628, 0.5700000000000001, 0.606, 0.442, 0.53, 0.482, 0.28600000000000003, 0.244, 0.25, 0.218, 0.178, 0.24, 0.232, 0.232], 'level_4': [0.604, 0.634, 0.646, 0.646, 0.634, 0.62, 0.658, 0.634, 0.596, 0.614, 0.544, 0.398, 0.33, 0.362, 0.256, 0.28, 0.194, 0.232, 0.212, 0.232, 0.232], 'level_5': [0.596, 0.65, 0.642, 0.652, 0.642, 0.636, 0.644, 0.596, 0.6, 0.5740000000000001, 0.5680000000000001, 0.432, 0.436, 0.308, 0.28400000000000003, 0.258, 0.226, 0.232, 0.24, 0.192, 0.212]}]}, 'run_2': {'synflow': [{'level_1': [0.616, 0.63, 0.624, 0.62, 0.636, 0.628, 0.6, 0.55, 0.516, 0.494, 0.47400000000000003, 0.34800000000000003, 0.244, 0.234, 0.212, 0.224, 0.24, 0.244, 0.244, 0.212, 0.24], 'level_2': [0.614, 0.63, 0.628, 0.624, 0.626, 0.626, 0.638, 0.59, 0.612, 0.454, 0.456, 0.512, 0.358, 0.258, 0.216, 0.258, 0.24, 0.212, 0.24, 0.212, 0.212], 'level_3': [0.61, 0.646, 0.646, 0.644, 0.642, 0.63, 0.638, 0.622, 0.552, 0.508, 0.536, 0.47000000000000003, 0.368, 0.258, 0.26, 0.23600000000000002, 0.234, 0.25, 0.234, 0.212, 0.24], 'level_4': [0.616, 0.634, 0.65, 0.652, 0.63, 0.638, 0.612, 0.624, 0.578, 0.5700000000000001, 0.586, 0.388, 0.362, 0.342, 0.28800000000000003, 0.26, 0.212, 0.244, 0.212, 0.24, 0.218], 'level_5': [0.612, 0.636, 0.634, 0.666, 0.652, 0.64, 0.634, 0.59, 0.606, 0.598, 0.526, 0.432, 0.41600000000000004, 0.262, 0.262, 0.228, 0.234, 0.22, 0.244, 0.252, 0.258]}]}}


for strategy in prune_strategy:

    level_data = {}
    for level in level_list:
        level_data['level_'+str(level)] = []

    for run in range(data['number_of_runs']):
        for level in level_list:
            level_data['level_'+str(level)].append(data['run_'+str(run)][strategy][0]['level_'+str(level)])

    for level in level_list:
        level_data['level_'+str(level)+'_average'], level_data['level_'+str(level)+'_min'], level_data['level_'+str(level)+'_max'] = calculate(level_data['level_'+str(level)])



    plt.rc('font', family='serif', size=12)
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')

    fig = plt.figure(figsize=(6, 6))


    ax = fig.add_subplot(1, 1, 1)

    for level in [1,2,3,4,5]:
        ax.plot(ratios, level_data['level_'+str(level)+'_average'], color= plt.cm.tab20b(level-1), ls='solid', linewidth = 2, marker = "o", label = "Iteration "+str(level))


    ax.set_facecolor("whitesmoke")
    plt.grid(color = 'gray', linestyle = 'solid', linewidth = 0.5)

    title_strategy_list = ['SF']
    dataset = 'Citeseer'
    title = title_strategy_list[prune_strategy.index(strategy)]+' Multiple Shot Weight Pruning ('+dataset+')'
    ax.set_title(title)
    ax.set_xlabel('Compression Ratio')
    ax.set_ylabel('Top-1 Accuracy')
    ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
    ax.set_xticks((0, 1, 2, 3, 4, 5))
    plt.legend(loc = 1, prop={'size':12})
    plt.savefig(title + '.png')
    plt.show()
