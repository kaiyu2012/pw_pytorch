import numpy as np
ratios = np.array(['0.0','0.25','0.5','0.75','1.0','1.25','1.5','1.75','2.0','2.25','2.5','2.75','3.0','3.25','3.5','3.75','4.0','4.25','4.5','4.75','5.0'])
#cora dataset
#best val acc
synflow_run1 = np.array(['0.7320', '0.7080', '0.7080', '0.7080', '0.7080', '0.7080', '0.7080', '0.7080', '0.7080', '0.7080', '0.7080', '0.7280', '0.6880', '0.6420', '0.6400', '0.6420', '0.6260', '0.5620', '0.5840', '0.5920', '0.5920'])


#prune_Adj percent in %
prune_adj_percent = np.array(['0.00', '26.32', '26.32', '26.32', '26.32', '26.32', '26.32', '26.32', '26.32', '26.32', '26.37', '43.68', '66.10', '81.31', '88.85', '93.44', '96.32', '98.13', '99.12', '99.73', '99.94'])

#global sparsity for adj in %
global_sparsity = np.array(['0.00', '8.26', '8.26', '8.26', '8.26', '8.26', '8.26', '8.26', '8.26', '8.26', '8.28', '13.72', '20.75', '25.53', '27.90', '29.34', '30.24', '30.81', '31.12', '31.32', '31.38'])


#pubmed dataset
#best val acc
p_synflow_run1 = np.array([])


#prune_Adj percent in %
p_prune_adj_percent = np.array([])

#global sparsity for adj in %
p_global_sparsity = np.array([])

#citeseer dataset
#best val acc
c_synflow_run1 = np.array(['0.7060', '0.6740', '0.6740', '0.6740', '0.6740', '0.6740', '0.6740', '0.6740', '0.6740', '0.6740', '0.6740', '0.6620', '0.6780', '0.6120', '0.6100', '0.5780', '0.5740', '0.5900', '0.5600', '0.5500', '0.5500'])


#prune_Adj percent in %
c_prune_adj_percent = np.array(['0.00', '48.13', '48.13', '48.13', '48.13', '48.13', '48.13', '48.13', '48.13', '48.13', '48.13', '54.85', '69.80', '81.73', '87.31', '92.41', '96.23', '98.49', '99.73', '100.00', '100.00'])

#global sparsity for adj in %
c_global_sparsity = np.array(['0.00', '6.47', '6.47', '6.47', '6.47', '6.47', '6.47', '6.47', '6.47', '6.47', '6.47', '7.38', '9.39', '11.00', '11.75', '12.43', '12.95', '13.25', '13.42', '13.45', '13.45'])