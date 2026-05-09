import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

#Load data
df = pd.read_csv("StudentsPerformance.csv")

'''#Check
print(df.head(3))
print(df.shape)
print(df.info())
print(df.isnull().sum())'''

#Create target variable
df["pass_fail"] = (df["math score"] >= 50).astype(int)

le = LabelEncoder()
categorical_columns = ["gender","race/ethnicity","parental level of education",
                       "lunch","test preparation course"]

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

#Feature and Target
x = df.drop(["math score","pass_fail"], axis = 1)
y = df["pass_fail"]

#Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#Training Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

print("="*40)
for name, model in models.items():
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    acc =accuracy_score(y_test,y_pred)
    print(f"{name}: {acc * 100:.2f}% accuracy")
print("="*40)

import matplotlib.pyplot as plt

# Best model - Logistic Regression
best_model = models["Random Forest"]
y_pred_best = best_model.predict(x_test)

# Classification Report
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_best))

# Feature Importance
features = x.columns
importances = best_model.feature_importances_

plt.figure(figsize=(8, 5))
plt.barh(features, importances, color="steelblue")
plt.xlabel("Importance Score")
plt.title("Feature Importance - Student Performance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
print("Chart saved as feature_importance.png")