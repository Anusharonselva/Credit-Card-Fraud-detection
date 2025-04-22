# Credit-Card-Fraud-detection
## Project Objective:
The goal of this project is to identify suspicious credit card transactions using a collection of anonymized transaction data. The challenge here is working with extremely imbalanced data — in which fraudulent transactions are extremely uncommon — and constructing a model capable of detecting such anomalies.
## Technologies used :
**Python**
**Pandas** for data manipulation
**Numpy** for numerical computation
**Scikit-learn** for machine learning algorithms
**Matplotlib/Seaborn** for data visualization
## code:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("/content/fraudTest.csv (1).zip")

print(df.head())
print(df.info())
print(df['is_fraud'].value_counts())

df = df.dropna()

df = pd.get_dummies(df, columns=['category'], drop_first=True)

df = df.drop(['trans_date_trans_time', 'merchant', 'first', 'last', 'street', 'city', 'state', 'job', 'dob', 'unix_time', 'trans_num', 'gender'], axis=1)

X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\nLogistic Regression Results:")
print(classification_report(y_test, y_pred_lr))
print("Accuracy:", accuracy_score(y_test, y_pred_lr))

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Results:")
print(classification_report(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

conf_matrix = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

```
## Output:
![Screenshot 2025-04-22 172557](https://github.com/user-attachments/assets/886540a0-62fa-4a31-a260-c92e28036b14)
## Colab link:
```https://colab.research.google.com/drive/1UcartrGKPb4IOSUIsCIi3JSYd_MOXBc7?usp=sharing```
