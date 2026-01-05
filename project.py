# ================================
# 1. IMPORT REQUIRED LIBRARIES
# ================================
import sqlite3
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ================================
# 2. CREATE & CONNECT DATABASE
# ================================
conn = sqlite3.connect("flowers.db")
cursor = conn.cursor()

# ================================
# 3. CREATE TABLE
# ================================
cursor.execute("""
CREATE TABLE IF NOT EXISTS iris_data (
    sepal_length REAL,
    sepal_width REAL,
    petal_length REAL,
    petal_width REAL,
    species INTEGER
)
""")

# ================================
# 4. LOAD IRIS DATASET
# ================================
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# ================================
# 5. INSERT DATA INTO DATABASE
# ================================
cursor.execute("DELETE FROM iris_data")

for i in range(len(X)):
    cursor.execute(
        "INSERT INTO iris_data VALUES (?, ?, ?, ?, ?)",
        (X[i][0], X[i][1], X[i][2], X[i][3], int(y[i]))
    )

conn.commit()
print("Data inserted into database successfully!")

# ================================
# 6. FETCH DATA FROM DATABASE
# ================================
df = pd.read_sql_query("SELECT * FROM iris_data", conn)
print("\nSample data from database:")
print(df.head())

# ================================
# 7. SPLIT DATA
# ================================
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# 8. MACHINE LEARNING MODEL
# ================================
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# ================================
# 9. MODEL PREDICTION
# ================================
y_pred = model.predict(X_test)

# ================================
# 10. MODEL EVALUATION
# ================================
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ================================
# 11. PREDICT NEW FLOWER
# ================================
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(new_flower)
predicted_species = target_names[prediction[0]]

print("\nNew Flower Data:", new_flower)
print("Predicted Species:", predicted_species)

# ================================
# 12. CLOSE DATABASE
# ================================
conn.close()
