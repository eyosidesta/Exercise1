import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

class MushroomClassifier:
    def __init__(self, data, target_col):
        self.data = data
        self.target_col = target_col
        self.features = [col for col in data.columns if col != target_col]
        self.label_encoder = LabelEncoder()

    def preprocess_data(self):
        # Encode the target variable
        self.data[self.target_col] = self.label_encoder.fit_transform(self.data[self.target_col])

        # Encode categorical features
        for feature in self.features:
            self.data[feature] = self.label_encoder.fit_transform(self.data[feature])

    def split_data(self):
        X = self.data[self.features]
        y = self.data[self.target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def build_decision_tree(self):
        self.model = DecisionTreeClassifier()
        self.model.fit(self.X_train, self.y_train)

    def visualize_decision_tree(self):
        plt.figure(figsize=(12, 8))
        plot_tree(self.model, feature_names=self.features, class_names=self.label_encoder.classes_, filled=True, rounded=True)
        plt.show()

    def run(self):
        self.preprocess_data()
        self.split_data()
        self.build_decision_tree()
        self.visualize_decision_tree()

# Provided data
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

# Create and run the classifier
mushroom_classifier = MushroomClassifier(df, target_col='class')
mushroom_classifier.run()


