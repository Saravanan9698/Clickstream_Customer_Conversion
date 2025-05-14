import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder

class MentalHealthPreprocessor:

    def __init__(self):

        self.encoders = {}  
        self.columns_to_drop = []  
        self.numerical_features = []
        self.categorical_features = []
        
    
    def fit(self, data):
        data = data.copy()
        
        self.numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = data.select_dtypes(include=['object']).columns.tolist()

        corr_matrix = data[self.numerical_features].corr().abs()

        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        self.columns_to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.6)]

        print(f"Columns Dropped Due to High Correlation: {self.columns_to_drop}")

        data = data.drop(columns=self.columns_to_drop, errors='ignore')

        self.numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

        plt.figure(figsize=(14, 6))
        sns.heatmap(data[self.numerical_features].corr(), cmap='turbo', annot=True, fmt='.2f', linewidths=0.5)
        plt.title("Post-Drop Correlation Heatmap")
        plt.show()

        for col in self.categorical_features:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                self.encoders[col] = le  
        print("Encoders:", self.encoders)

        return data

    def transform(self, data):

        data = data.drop(columns=self.columns_to_drop, errors='ignore')

        for col in self.categorical_features:
            if col in data.columns and col in self.encoders:
                data[col] = self.encoders[col].transform(data[col].astype(str))

        return data

file_path = r'D:\Projects\Mini_Projects\Mental_Health_Survey\Research_Data\cleaned_data.csv'
data = pd.read_csv(file_path)

preprocessor = MentalHealthPreprocessor()
preprocessed_data = preprocessor.fit(data)

with open(r'D:\Projects\Mini_Projects\Mental_Health_Survey\Model\preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

preprocessed_data.to_csv(r'D:\Projects\Mini_Projects\Mental_Health_Survey\Research_Data\preprocessed_data.csv', index=False)

print("Preprocessing completed & saved as preprocessor.pkl")