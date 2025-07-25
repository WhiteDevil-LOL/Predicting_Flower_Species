import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix

iris = load_iris()
X = pd.DataFrame(iris.data , columns=iris.feature_names)
y = pd.Series(iris.target)

species_names = iris.target_names
print("Features (X) Head:")
print(X.head())
print("\nTarget (y) Head (numerical labels):")
print(y.head())
print("\nSpecies Names (mapping):")
for i, name in enumerate(species_names):
    print(f"{i}: {name}")

print("\nDataset Info:")
X.info()
print("\nDataset Description:")
print(X.describe())

df_full = X.copy()
df_full['species'] = y
df_full['species_name'] = df_full['species'].map({i : name for i , name in enumerate(species_names)})
print("\n--- Generating Pair Plot ---\n")
sns.pairplot(df_full, hue='species_name', diag_kind='kde')
plt.suptitle('Pairplot of IRIS Features by Species', y = 1.02)
plt.show()

plt.figure(figsize=(10,8))
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='species_name', data=df_full, palette='plasma', s=100)
plt.title('Sepal Length vs. Sepal Width by Species')
plt.xlabel('Sepal-Length(cm)')
plt.ylabel('Sepal-Width(cm)')
plt.legend(title = 'Species')
plt.grid(True , linestyle = "--", alpha = 0.7)
plt.show()

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 1 , stratify = y)
print(f"Training Set Size: {len(X_train)}")
print(f"Testing Set Size: {len(X_test)}")
print(f"Species Distribution in y_train:\n {y_train.value_counts(normalize = True)}")
print(f"Species Distribution in y_test:\n {y_test.value_counts(normalize = True)}")


model = KNeighborsClassifier(n_neighbors = 5)
print("--- TRAINING MODEL ---")
model.fit(X_train , y_train)
print("--- MODEL TRAINING COMPLETED ---")
y_pred = model.predict(X_test)
print("Actual Species Labels: ")
print(y_test)
print("Predicted Species Labels: ")
print(y_pred)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on Test set: {accuracy:.4f}")
print("Classification Report: ")
print(classification_report(y_test , y_pred , target_names = species_names))
conf_matrix = confusion_matrix(y_test , y_pred)
print("Confusion MAtrix: ")
print(conf_matrix)


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=species_names, yticklabels=species_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


new_flower_measurements = np.array([[6.0, 2.7, 4.2, 1.3]]) # Example: might be Versicolor
predicted_species_numerical = model.predict(new_flower_measurements)
predicted_species_name = species_names[predicted_species_numerical[0]]

print(f"\nMeasurements for another new flower: {new_flower_measurements[0]}")
print(f"Predicted species name: {predicted_species_name}")