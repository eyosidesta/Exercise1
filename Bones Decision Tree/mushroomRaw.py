from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import pandas as pd

data = {
    'cap': ['convex', 'flat', 'convex', 'convex', 'flat'],
    'scale': ['smooth', 'scales', 'smooth', 'scales', 'scales'],
    'gills': ['free', 'attached', 'free', 'free', 'attached'],
    'tubes': ['no', 'yes', 'no', 'no', 'yes'],
    'pores': ['no', 'no', 'no', 'no', 'no'],
    'ring': ['no', 'no', 'yes', 'no', 'no'],
    'stipe': ['equal', 'equal', 'equal', 'equal', 'equal'],
    'stalk': ['tapering', 'tapering', 'enlarging', 'tapering', 'enlarging'],
    'volva': ['no', 'no', 'yes', 'no', 'no'],
    'scales': ['no', 'yes', 'no', 'no', 'no'],
    'class': ['poisonous', 'safe', 'poisonous', 'poisonous', 'safe']
}

df = pd.DataFrame(data)

# Encode categorical features
df_encoded = pd.get_dummies(df.drop(columns=['class']))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df_encoded, df['class'], test_size=0.2, random_state=42)

# Create and fit the Decision Tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

plt.figure(figsize=(12, 8))
plot_tree(dt_model, feature_names=df_encoded.columns, class_names=df['class'].unique(), filled=True, rounded=True)
plt.show()
