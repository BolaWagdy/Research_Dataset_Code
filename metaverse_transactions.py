import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv("Research Dataset\metaverse_transactions_dataset.csv")

# Convert timestamp to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Encode categorical features
label_encoders = {}
categorical_columns = ['transaction_type', 'location_region', 'purchase_pattern', 'age_group', 'anomaly']
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders for inverse transformation if needed

# Data Visualization
plt.figure(figsize=(12, 5))
sns.countplot(x=df['transaction_type'])
plt.title("Transaction Type Distribution")
plt.xticks(rotation=45)
plt.show()

# Correlation Heatmap
numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

# Machine Learning Model for Anomaly Detection
X = df[['amount', 'transaction_type', 'login_frequency', 'session_duration', 'risk_score']]
y = df['anomaly']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
