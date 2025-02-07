
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = np.array(iris.data)  # Features: Sepal length, Sepal width, Petal length, Petal width
Y = np.array(iris.target)  # Target species (0 = setosa, 1 = versicolor, 2 = virginica)

# Feature names for reference
feature_names = iris.feature_names

# Step 1: Calculate statistical measures
mean_values = np.mean(X, axis=0)  # Mean for each feature
median_values = np.median(X, axis=0)  # Median for each feature
std_dev_values = np.std(X, axis=0)  # Standard deviation for each feature
min_values = np.min(X, axis=0)  # Minimum values for each feature
max_values = np.max(X, axis=0)  # Maximum values for each feature

# Display statistics
print("\nIris Dataset Statistics:")
for i in range(X.shape[1]):
    print(f"{feature_names[i]}: Mean = {mean_values[i]:.2f}, Median = {median_values[i]:.2f}, "
          f"Std Dev = {std_dev_values[i]:.2f}, Min = {min_values[i]:.2f}, Max = {max_values[i]:.2f}")

# Step 2: Extract Sepal Length and Sepal Width
sepal_data = X[:, :2]  # Extract first two columns (Sepal Length & Sepal Width)

# Step 3: Data Visualization with Matplotlib
plt.figure(figsize=(15, 5))

# Scatter Plot: Sepal Length vs Sepal Width
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap="viridis", edgecolors="k", alpha=0.7)
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Sepal Length vs Sepal Width")
plt.colorbar(label="Species")

# Histogram: Distribution of Sepal Length
plt.subplot(1, 3, 2)
plt.hist(X[:, 0], bins=20, color="blue", edgecolor="black", alpha=0.7)
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.title("Histogram of Sepal Length")

# Line Plot: Petal Length vs Petal Width
plt.subplot(1, 3, 3)
plt.plot(X[:, 2], X[:, 3], marker="o", linestyle="-", color="red", alpha=0.7)
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Petal Length vs Petal Width")

# Display all plots
plt.tight_layout()
plt.show()