import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = np.loadtxt('/CS170_Proj_2/small-test-dataset-1.txt')
columns_to_normalize = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

scaler = MinMaxScaler()

data[:, columns_to_normalize] = scaler.fit_transform(data[:, columns_to_normalize])

print(data)

def nearest_neighbor(train_data, test_instance):
    
    y_train = train_data[:, 0]
    X_train = train_data[:, 1:]

    distances = np.linalg.norm(X_train - test_instance, axis=1)

    nearest_neighbor_index = np.argmin(distances)

    return y_train[nearest_neighbor_index]



test_instance = np.array([0.06883, 0.740, 0.289108, 0.763421, 0.72686, 0.9103, 0.602714, 0.3621, 0.02531, 0.8503])

# Classify the test instance using Nearest Neighbor
label = nearest_neighbor(data, test_instance)
print(f"The predicted label for the test instance is: {label}")