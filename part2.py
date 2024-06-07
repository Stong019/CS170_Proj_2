import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = np.loadtxt('/CS170_Proj_2/small-test-dataset-1.txt')
columns_to_normalize = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

scaler = MinMaxScaler()

data[:, columns_to_normalize] = scaler.fit_transform(data[:, columns_to_normalize])

print(data)