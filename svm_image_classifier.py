import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Dataset path
DATADIR = "dataset"

CATEGORIES = ["cats", "dogs"]

data = []
labels = []

IMG_SIZE = 64

# Load images
for category in CATEGORIES:

    path = os.path.join(DATADIR, category)
    label = CATEGORIES.index(category)

    for img in os.listdir(path):

        try:
            img_path = os.path.join(path, img)

            image = cv2.imread(img_path)

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

            data.append(image.flatten())

            labels.append(label)

        except Exception as e:
            pass

# Convert to numpy arrays
X = np.array(data)
y = np.array(labels)

print("Dataset Loaded Successfully!")
print("Total Images:", len(X))

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create SVM model
model = SVC(kernel='linear')

# Train model
print("Training SVM Model...")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Show sample predictions
plt.figure(figsize=(10,5))

for i in range(6):

    plt.subplot(2,3,i+1)

    image = X_test[i].reshape(IMG_SIZE, IMG_SIZE, 3)

    plt.imshow(cv2.cvtColor(image.astype('uint8'), cv2.COLOR_BGR2RGB))

    prediction = "Dog" if y_pred[i] == 1 else "Cat"

    plt.title(f"Predicted: {prediction}")

    plt.axis("off")

# Create screenshots folder
os.makedirs("screenshots", exist_ok=True)

# Save output
plt.savefig("screenshots/output.png")

plt.show()