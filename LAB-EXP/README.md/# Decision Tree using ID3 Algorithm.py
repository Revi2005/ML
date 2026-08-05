# Decision Tree using ID3 Algorithm

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

# -------------------------------
# Create Dataset
# -------------------------------
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain',
                'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain',
                'Sunny', 'Overcast', 'Overcast', 'Rain'],

    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool',
                    'Cool', 'Cool', 'Mild', 'Cool', 'Mild',
                    'Mild', 'Mild', 'Hot', 'Mild'],

    'Humidity': ['High', 'High', 'High', 'High', 'Normal',
                 'Normal', 'Normal', 'High', 'Normal', 'Normal',
                 'Normal', 'High', 'Normal', 'High'],

    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak',
             'Strong', 'Strong', 'Weak', 'Weak', 'Weak',
             'Strong', 'Strong', 'Weak', 'Strong'],

    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes',
             'No', 'Yes', 'No', 'Yes', 'Yes',
             'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

print("Dataset:\n")
print(df)

# -------------------------------
# Encode Categorical Data
# -------------------------------
le = LabelEncoder()

df_encoded = df.copy()

for column in df.columns:
    df_encoded[column] = le.fit_transform(df[column])

X = df_encoded.iloc[:, :-1]
y = df_encoded.iloc[:, -1]

# -------------------------------
# Train ID3 Decision Tree
# -------------------------------
model = DecisionTreeClassifier(criterion='entropy')

model.fit(X, y)

# -------------------------------
# Display Decision Tree
# -------------------------------
print("\nDecision Tree Rules:\n")
print(export_text(model, feature_names=list(X.columns)))

# -------------------------------
# Classify New Sample
# -------------------------------

new_sample = {
    'Outlook': 'Sunny',
    'Temperature': 'Cool',
    'Humidity': 'High',
    'Wind': 'Strong'
}

# Encode new sample
encoded_sample = []

for col in X.columns:
    encoder = LabelEncoder()
    encoder.fit(df[col])
    encoded_sample.append(encoder.transform([new_sample[col]])[0])

prediction = model.predict([encoded_sample])

print("\nNew Sample:")
print(new_sample)

if prediction[0] == 1:
    print("\nPrediction: Play = Yes")
else:
    print("\nPrediction: Play = No")