import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = np.loadtxt('/content/small-test-dataset.txt')
columns_to_normalize = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

scaler = MinMaxScaler()

data[:, columns_to_normalize] = scaler.fit_transform(data[:, columns_to_normalize])

#test_instance = np.array([0.06883, 0.740, 0.289108, 0.763421, 0.72686, 0.9103, 0.602714, 0.3621, 0.02531, 0.8503])

# Classify the test instance using Nearest Neighbor
#label = nearest_neighbor(data, test_instance)
#print(f"The predicted label for the test instance is: {label}")


class LeaveOneOutValidator:
    def __init__(self, dataset, classifier):
        self.dataset = dataset
        self.classifier = classifier
    
    def validate(self, feature_subset):
        correct_predictions = 0
        n_samples = self.dataset.shape[0]
        
        for i in range(n_samples):
            # Leave one out
            train_data = np.delete(self.dataset, i, axis=0)
            test_instance = self.dataset[i, :]
            
            # Extract the features for train and test based on the subset
            X_train = train_data[:, feature_subset]
            y_train = train_data[:, 0]
            X_test = test_instance[feature_subset]
            y_test = test_instance[0]
            
            # Create a new training set with the selected features
            new_train_data = np.column_stack((y_train, X_train))
            
            # Predict the label using the classifier
            predicted_label = self.classifier(new_train_data, X_test)
            
            # Check if the prediction is correct
            if predicted_label == y_test:
                correct_predictions += 1
        
        # Calculate the accuracy
        accuracy = correct_predictions / n_samples
        return accuracy



def nearest_neighbor(train_data, test_instance):
    y_train = train_data[:, 0]
    X_train = train_data[:, 1:]
    distances = np.linalg.norm(X_train - test_instance, axis=1)
    nearest_neighbor_index = np.argmin(distances)
    return y_train[nearest_neighbor_index]

# Instantiate the validator with the dataset and classifier
#validator = LeaveOneOutValidator(data, nearest_neighbor)

# Define the feature subset (excluding the label column)
#feature_subset = [ 3, 5, 7]

# Calculate and print the accuracy
#accuracy = validator.validate(feature_subset)

import math
import random


def ForwardSelection(num_features, validator):
    selectedFeatures = []
    best_accuracy = -1
    print("\nBeginning Search")
    
    for i in range(1, num_features + 1):
        bestFeature = None
        for currentFeature in range(1, num_features + 1):
            if currentFeature not in selectedFeatures:
                current_set = selectedFeatures + [currentFeature]
                accuracy = validator.validate(current_set)
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

def BackwardElimination(num_features, validator):
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
            accuracy = validator.validate(current_set)
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
    validator = LeaveOneOutValidator(data, nearest_neighbor)
    print('Welcome to Group 61\'s Feature Selection Algorithm.')

    num_features = int(input('Please enter total number of features: '))

    algo = int(input('Type the number of algorithm you want to run. \n'
                     '\n1) Forward Selection'
                     '\n2) Backward Elimination\n\n'))
    
    if (algo == 1):
        ForwardSelection(num_features, validator)
    elif (algo == 2):
        BackwardElimination(num_features, validator)
    print(validator.validate([3, 5, 7]))

if __name__ == "__main__":
    main()