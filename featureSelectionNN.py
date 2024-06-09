import math
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def LeaveOneOutValidator(dataset, classifier, feature_subset):
    correct_predictions = 0
    n_samples = dataset.shape[0]
    
    for i in range(n_samples):
        train_data = np.delete(dataset, i, axis=0)
        test_instance = dataset[i, :]
        
        X_train = train_data[:, feature_subset]
        y_train = train_data[:, 0]
        X_test = test_instance[feature_subset]
        y_test = test_instance[0]
        
        new_train_data = np.column_stack((y_train, X_train))
        
        predicted_label = classifier(new_train_data, X_test)
        
        if predicted_label == y_test:
            correct_predictions += 1
    
    accuracy = correct_predictions / n_samples
    return accuracy



def nearest_neighbor(train_data, test_instance):
    y_train = train_data[:, 0]
    X_train = train_data[:, 1:]
    distances = np.linalg.norm(X_train - test_instance, axis=1)
    nearest_neighbor_index = np.argmin(distances)
    return y_train[nearest_neighbor_index]

def ForwardSelection(num_features, dataset):
    selectedFeatures = []
    best_accuracy = -1
    print("\nBeginning Search")
    
    for i in range(1, num_features + 1):
        bestFeature = None
        for currentFeature in range(1, num_features + 1):
            if currentFeature not in selectedFeatures:
                current_set = selectedFeatures + [currentFeature]
                accuracy = LeaveOneOutValidator(dataset, nearest_neighbor, current_set)
                print(f"Using Features{current_set}, accuracy is {accuracy:.4f}")
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    bestFeature = currentFeature
        if bestFeature:
            selectedFeatures.append(bestFeature)
            print(f"Feature set {selectedFeatures} was best, accuracy is {best_accuracy:.4f}")
        else:
            break
    
    print(f"Finished search!!! the best feature subset is {selectedFeatures}, which has an accuracy of {best_accuracy:.4f}\n")

def BackwardElimination(num_features, dataset):
    selectedFeatures = list(range(1, num_features + 1))
    best_accuracy = -1
    print("\nBeginning Search")
    
    accuracy = random.uniform(0, 1)
    best_accuracy = accuracy
    print(f"Using Features{selectedFeatures}, accuracy is {accuracy:.4f}")

    for i in range(num_features, 0, -1):
        bestFeature = None
        for currentFeature in selectedFeatures:
            current_set = [feature for feature in selectedFeatures if feature != currentFeature]
            accuracy = LeaveOneOutValidator(dataset, nearest_neighbor, current_set)
            print(f"Using Features{current_set}, accuracy is {accuracy:.4f}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                bestFeature = currentFeature
        if bestFeature:
            selectedFeatures.remove(bestFeature)
            print(f"Feature set {selectedFeatures} was best, accuracy is {best_accuracy:.4f}")
        else:
            break
    
    print(f"Finished search!!! the best feature subset is {selectedFeatures}, which has an accuracy of {best_accuracy:.4f}\n")

def main():

    print('Welcome to Group 61\'s Feature Selection Algorithm.')

    data = np.loadtxt('/content/large-test-dataset-1.txt')
    columns_to_normalize = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    scaler = MinMaxScaler()
    data[:, columns_to_normalize] = scaler.fit_transform(data[:, columns_to_normalize])

    num_features = data.shape[1] - 1

    algo = int(input('Type the number of algorithm you want to run. \n'
                     '\n1) Forward Selection'
                     '\n2) Backward Elimination\n\n'))
    
    if (algo == 1):
        ForwardSelection(num_features, data)
    elif (algo == 2):
        BackwardElimination(num_features, data)

if __name__ == "__main__":
    main()