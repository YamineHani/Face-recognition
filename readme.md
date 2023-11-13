# Facial Recognition

This Python script performs facial recognition using Principal Component Analysis (PCA) and the k-Nearest Neighbors (KNN) algorithm. It uses the scikit-learn library for machine learning tasks.

## Requirements

Make sure you have the following libraries installed:

- os
- cv2 (OpenCV)
- numpy
- sklearn
- matplotlib

You can install them using the following:

```bash
pip install opencv-python numpy scikit-learn matplotlib
```

## Description

The script performs the following tasks:

1. **Load Data:**
   - Reads images from a specified directory (`'archive'` in this case).
   - Converts images to grayscale.
   - Flattens the image matrices to create a data matrix and corresponding label vector.

```python
dataset_dir = 'archive'
data_matrix, label_vector = load_data(dataset_dir)
```

2. **Split Data:**
   - Splits the data into training and testing sets using a 70-30 split ratio.

```python
D_train, D_test, y_train, y_test = split_data(data_matrix, label_vector, 0.3)
```

3. **Perform PCA and KNN:**
   - Performs PCA on the training set and applies KNN for various alpha and k values.
   - Prints the accuracy, k value, and the number of principal components (R) for each combination.

```python
alpha_values = [0.95, 0.99]
k_values = [1, 3, 5]
accuracy_dict = perform_pca_knn(D_train, D_test, y_train, y_test, alpha_values, k_values)
```

4. **Plot Results:**
   - Plots accuracy vs. alpha values for different k values.

```python
plot_accuracy_vs_alpha(alpha_values, k_values, accuracy_dict)
```

   - Plots accuracy vs. k values for different alpha values.

```python
plot_accuracy_vs_k(alpha_values, k_values, accuracy_dict)
```

