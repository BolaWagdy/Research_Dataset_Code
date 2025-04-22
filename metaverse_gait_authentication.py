import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("Research Dataset\metaverse_gait_authentication_dataset.csv")

# Normalize numerical features
scaler = StandardScaler()
numeric_columns = df.columns[:-1]  # All columns except 'Label'
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Exploratory Data Analysis
plt.figure(figsize=(12, 5))
sns.countplot(x=df['Label'])
plt.title("Distribution of Gait Labels")
plt.xticks(rotation=45)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

# Machine Learning Model
X = df.iloc[:, :-1]  # Features
y = df['Label']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))