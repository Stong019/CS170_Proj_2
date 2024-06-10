import math
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def leave_one_out_validator(dataset, classifier, feature_subset, k):
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

        neighbors = classifier(new_train_data, X_test, k)
        predicted_label = np.argmax(np.bincount(neighbors.astype(int)))

        if predicted_label == y_test:
            correct_predictions += 1

    accuracy = correct_predictions / n_samples
    return accuracy


def nearest_neighbors(train_data, test_instance, k) -> np.ndarray[np.float64]:
    y_train = train_data[:, 0]
    X_train = train_data[:, 1:]
    distances = np.linalg.norm(X_train - test_instance, axis=1)
    nearest_neighbor_indices = np.argsort(distances)[:k]
    return y_train[nearest_neighbor_indices]


def forward_selection(num_features, dataset, k=1) -> float:
    selected_features = []
    best_accuracy = -1
    print("\nBeginning Search")

    for i in range(1, num_features + 1):
        best_feature = None
        for current_feature in range(1, num_features + 1):
            if current_feature not in selected_features:
                current_set = selected_features + [current_feature]
                accuracy = leave_one_out_validator(
                    dataset, nearest_neighbors, current_set, k
                )
                print(f"Using Features{current_set}, accuracy is {accuracy:.4f}")
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_feature = current_feature
        if best_feature:
            selected_features.append(best_feature)
            print(
                f"Feature set {selected_features} was best, accuracy is {best_accuracy:.4f}"
            )
        else:
            break

    print(
        f"Finished search!!! the best feature subset is {selected_features}, which has an accuracy of {best_accuracy:.4f}\n"
    )
    return best_accuracy


def backward_elimination(num_features, dataset, k=1) -> float:
    selected_features = list(range(1, num_features + 1))
    best_accuracy = -1
    print("\nBeginning Search")

    accuracy = random.uniform(0, 1)
    best_accuracy = accuracy
    print(f"Using Features{selected_features}, accuracy is {accuracy:.4f}")

    for i in range(num_features, 0, -1):
        best_feature = None
        for current_feature in selected_features:
            current_set = [
                feature for feature in selected_features if feature != current_feature
            ]
            accuracy = leave_one_out_validator(
                dataset, nearest_neighbors, current_set, k
            )
            print(f"Using Features{current_set}, accuracy is {accuracy:.4f}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature = current_feature
        if best_feature:
            selected_features.remove(best_feature)
            print(
                f"Feature set {selected_features} was best, accuracy is {best_accuracy:.4f}"
            )
        else:
            break

    print(
        f"Finished search!!! the best feature subset is {selected_features}, which has an accuracy of {best_accuracy:.4f}\n"
    )
    return best_accuracy


def main():
    print("Welcome to Group 61's Feature Selection Algorithm.")

    data = np.loadtxt("./content/custom-small-test-dataset.txt")
    columns_to_normalize = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    scaler = MinMaxScaler()
    data[:, columns_to_normalize] = scaler.fit_transform(data[:, columns_to_normalize])

    num_features = data.shape[1] - 1

    algo = int(
        input(
            "Type the number of algorithm you want to run. \n"
            "\n1) Forward Selection"
            "\n2) Backward Elimination\n\n"
        )
    )

    if algo == 1:
        forward_selection(num_features, data)
    elif algo == 2:
        backward_elimination(num_features, data)


if __name__ == "__main__":
    main()
