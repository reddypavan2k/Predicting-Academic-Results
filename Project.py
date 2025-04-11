import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

class StudentPerformancePredictor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.label_encoders = {}
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Perceptron': Perceptron(),
            'Logistic Regression': LogisticRegression(),
            'MLP Classifier': MLPClassifier(activation='logistic', max_iter=1000)
        }
    
    def load_and_preprocess(self):
        self.data = pd.read_csv(self.file_path)
        drop_cols = [
            'gender', 'StageID', 'GradeID', 'NationalITy', 'PlaceofBirth',
            'SectionID', 'Topic', 'Semester', 'Relation',
            'ParentschoolSatisfaction', 'ParentAnsweringSurvey',
            'AnnouncementsView'
        ]
        self.data.drop(columns=drop_cols, inplace=True)

        for column in self.data.select_dtypes(include='object').columns:
            le = LabelEncoder()
            self.data[column] = le.fit_transform(self.data[column])
            self.label_encoders[column] = le

        self.data = shuffle(self.data, random_state=42)
        self.X = self.data.drop('Class', axis=1)
        self.y = self.data['Class']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )

    def train_models(self):
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            print(f"\n{name} Classification Report:")
            print(classification_report(self.y_test, y_pred))

    def plot_count(self, feature, hue=None, order=None, hue_order=None, figsize=(10,6)):
        plt.figure(figsize=figsize)
        sns.countplot(x=feature, hue=hue, data=self.data, order=order, hue_order=hue_order)
        plt.title(f'{feature} Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def predict_from_input(self):
        print("\n--- Predict Student Performance ---")
        try:
            raised_hands = int(input("Raised Hands: "))
            visited_resources = int(input("Visited Resources: "))
            discussion = int(input("Discussion: "))
            absence = input("Student Absence Days (Under-7/Above-7): ")
            absence_encoded = 1 if absence.strip().lower() == 'under-7' else 0

            input_data = np.array([[raised_hands, visited_resources, discussion, absence_encoded]])

            for name, model in self.models.items():
                prediction = model.predict(input_data)[0]
                if 'Class' in self.label_encoders:
                    prediction = self.label_encoders['Class'].inverse_transform([prediction])[0]
                print(f"{name} Prediction: {prediction}")
        except Exception as e:
            print(f"Error: {e}")

# ==== RUN SCRIPT ====
if __name__ == "__main__":
    predictor = StudentPerformancePredictor("AI-Data.csv")
    predictor.load_and_preprocess()
    predictor.train_models()
    
    # Optional Visualizations
    predictor.plot_count('Class', order=['L', 'M', 'H'])
    predictor.plot_count('gender', hue='Class', order=['M', 'F'], hue_order=['L', 'M', 'H'])
    
    # Prediction
    predictor.predict_from_input()
