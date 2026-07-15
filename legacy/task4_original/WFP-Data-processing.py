import numpy as np

# Path to your npz file
npz_file_path = "C:\\Users\\bingh\\Desktop\\MIL\\CIC-2018-csv\\2024DOQ\\doq_100_360_47500_4_ow.npz"

# Load the data
data = np.load(npz_file_path)
print("Keys in the npz file:", data.files)

# Extract the arrays
x_train = data["x_train"]
y_train = data["y_train"]
x_test = data["x_test"]
y_test = data["y_test"]

# Print data shapes
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

# Analyze labels
unique_train_labels = np.unique(y_train)
unique_test_labels = np.unique(y_test)

print("Number of unique labels in training set:", len(unique_train_labels))
print("Unique labels in training set:", unique_train_labels)
print("Number of unique labels in testing set:", len(unique_test_labels))
print("Unique labels in testing set:", unique_test_labels)

# Check if it's likely an open-world setting
# Open-world setups often have a separate "unknown" or "unmonitored" label.
# This is just a guess: if you know the label indexing scheme, adjust accordingly.
max_label_train = y_train.max()
max_label_test = y_test.max()
print("Maximum label in training set:", max_label_train)
print("Maximum label in testing set:", max_label_test)

# If you know the number of monitored classes, you can compare it with the label distribution.
# For now, this prints basic label statistics. You may need domain knowledge to confirm open-world settings.
