import math
import random

def ForwardSelection(num_features):
    selectedFeatures = []
    best_accuracy = -1
    print("\nBeginning Search")
    
    for i in range(1, num_features + 1):
        bestFeature = None
        for currentFeature in range(1, num_features + 1):
            if currentFeature not in selectedFeatures:
                current_set = selectedFeatures + [currentFeature]
                accuracy = random.uniform(0, 1)
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

def BackwardElimination(num_features):
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
            accuracy = random.uniform(0, 1)
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

    num_features = int(input('Please enter total number of features: '))

    algo = int(input('Type the number of algorithm you want to run. \n'
                     '\n1) Forward Selection'
                     '\n2) Backward Elimination\n\n'))
    
    if (algo == 1):
        ForwardSelection(num_features)
    elif (algo == 2):
        BackwardElimination(num_features)

if __name__ == "__main__":
    main()
