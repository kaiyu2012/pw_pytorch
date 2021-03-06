#prune_Epochs 10
#cora dataset
import numpy as np
ratios = np.array(['0.0','0.25','0.5','0.75','1.0','1.25','1.5','1.75','2.0','2.25','2.5','2.75','3.0','3.25','3.5','3.75','4.0','4.25','4.5','4.75','5.0'])
#best val acc
synflow_run1 = np.array(['0.7440', '0.7400', '0.7460', '0.7480', '0.7460', '0.7400', '0.7360', '0.6700', '0.5980', '0.5140', '0.2620', '0.2200', '0.2020', '0.1740', '0.2860', '0.3000', '0.3000', '0.3000', '0.3000', '0.3000', '0.3000'])


#weight sparsity in %
weight_sparsity = np.array(['0.00', '43.76', '68.38', '82.21', '90.00', '94.38', '96.84', '98.22', '99.00', '99.44', '99.68', '99.82', '99.90', '99.94', '99.97', '99.98', '99.99', '99.99', '100.00', '100.00', '100.00'])

#global sparsity in %
global_sparsity = np.array(['0.00', '29.99', '46.86', '56.34', '61.68', '64.68', '66.36', '67.31', '67.85', '68.15', '68.32', '68.41', '68.46', '68.49', '68.51', '68.52', '68.52', '68.53', '68.53', '68.53', '68.53'])


#pubmed dataset
#best val acc
p_synflow_run1 = np.array(['0.7860', '0.8000', '0.8020', '0.8040', '0.8060', '0.8000', '0.7860', '0.7580', '0.7320', '0.6780', '0.6140', '0.4060', '0.3820', '0.4160'. '0.4160', '0.4160', '0.4160', '0.4160', '0.4160', '0.4160', '0.4160'])

#weight sparsity in %
p_weight_sparsity = np.array(['0.00', '43.76', '68.36', '82.21', '90.00', '94.37', '96.83', '98.21', '98.99', '99.43', '99.68', '99.81', '99.89', '99.94', '99.96', '99.98', '99.99', '99.99', '99.99', '99.99', '99.99'])

#global sparsity in %
p_global_sparsity = np.array(['0.00', '3.64', '5.69', '6.84', '7.49', '7.85', '8.06', '8.17', '8.24', '8.27', '8.29', '8.31', '8.31', '8.32', '8.32', '8.32', '8.32', '8.32', '8.32', '8.32', '8.32'])


#citeseer dataset
#best val acc
c_synflow_run1 = np.array(['0.6020', '0.6080', '0.6080', '0.6060', '0.6060', '0.6000', '0.6160', '0.6060', '0.5900', '0.5800', '0.5800', '0.5180', '0.2600', '0.2560', '0.2600', '0.2660', '0.2560', '0.2180', '0.2160', '0.2160', '0.2120'])

#weight sparsity in %
c_weight_sparsity = np.array(['0.00', '44.71', '68.38', '82.22', '90.00', '94.38', '96.84', '98.22', '99.00', '99.44', '99.68', '99.82', '99.90', '99.94', '99.97', '99.98', '99.99', '99.99', '100.00', '100.00', '100.00'])

#global sparsity in %
c_global_sparsity = np.array(['0.00', '38.68', '59.16', '71.13', '77.86', '81.65', '83.78', '84.98', '85.65', '86.03', '86.24', '86.36', '86.43', '86.47', '86.49', '86.50', '86.51', '86.51', '86.51', '86.51', '86.51'])