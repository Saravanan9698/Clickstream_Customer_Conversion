import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

file_path = "D:\\Projects\\Mini_Projects\\Mental_Health_Survey\\Research_Data\\train.csv"

data = pd.read_csv(file_path)

data.drop(columns=['id', 'Name', 'Depression'], axis=1, inplace=True)

data['Age'] = data['Age'].astype(int)

def check_city(city_name):

    city_list = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Ahmedabad", "Chennai", "Kolkata", "Pune", "Jaipur", "Lucknow",
                 "Kanpur", "Nagpur", "Indore", "Thane", "Bhopal", "Visakhapatnam", "Patna", "Vadodara", "Vasai-Virar", "Ghaziabad", "Ludhiana",
                 "Agra", "Nashik", "Faridabad", "Meerut", "Rajkot", "Varanasi", "Srinagar", "Aurangabad", "Dhanbad", "Amritsar", "Kalyan"]
    return city_name if city_name in city_list else "Invalid"

data['City'] = data['City'].apply(check_city)
data = data[data['City'] != 'Invalid']

unwanted_professions = [
    'Student', 'Academic', 'Profession', 'Yogesh', 'BCA', 'Unemployed', 'PhD',
    'MBA', 'LLM', 'Unveil', 'B.Com', 'FamilyVirar', 'City Manager', 'BBA',
    'Medical Doctor', 'Working Professional', 'MBBS', 'Patna', 'Pranav', 'B.Ed',
    'Nagpur', 'Moderate', 'M.Ed', 'Visakhapatnam', 'BE', 'Yuvraj'
]
data = data[~data['Profession'].isin(unwanted_professions)]

data['Profession'].fillna('Not Yet Disclosed', inplace=True)
data['Academic Pressure'].fillna(0, inplace=True)
data['Academic Pressure'] = data['Academic Pressure'].astype(int)
data['Work Pressure'].fillna(0, inplace=True)
data['Work Pressure'] = data['Work Pressure'].astype(int)
data['CGPA'].fillna(0, inplace=True)
data['Study Satisfaction'].fillna(0, inplace=True)
data['Study Satisfaction'] = data['Study Satisfaction'].astype(int)
data['Job Satisfaction'].fillna(0, inplace=True)
data['Job Satisfaction'] = data['Job Satisfaction'].astype(int)

def categorize_sleep(duration):

    duration = duration.lower()
    if any(x in duration for x in ["less than 5", "1-2", "2-3", "3-4", "4-5", "than 5"]):
        return "Less than 5 hours"
    elif any(x in duration for x in ["5-6", "4-6", "3-6", "6-7", "1-6"]):
        return "5-6 hours"
    elif any(x in duration for x in ["7-8", "6-8", "8 hours"]):
        return "6-8 hours"
    elif any(x in duration for x in ["more than 8", "8-9", "9-11", "10-11"]):
        return "More than 8 hours"
    else:
        return "Invalid"

data['Sleep Duration'] = data['Sleep Duration'].apply(categorize_sleep)
data = data[data['Sleep Duration'] != 'Invalid']

def categorize_diet(habit):

    if pd.isna(habit):
        return "Invalid"
    habit = str(habit).lower().strip()
    words = habit.split()
    if any(x in words for x in ["healthy", "more healthy"]):
        return "Healthy"
    elif "moderate" in words:
        return "Moderate"
    elif any(x in words for x in ["unhealthy", "less healthy", "less than healthy", "no healthy"]):
        return "Unhealthy"
    else:
        return "Invalid"

data['Dietary Habits'] = data['Dietary Habits'].apply(categorize_diet)
data = data[data['Dietary Habits'] != 'Invalid']

top_28_degrees = data['Degree'].value_counts().head(28).index
data = data[data['Degree'].isin(top_28_degrees)]

data['Work/Study Hours'] = data['Work/Study Hours'].astype(int)
data['Financial Stress'].fillna(data['Financial Stress'].median(), inplace=True)
data['Financial Stress'] = data['Financial Stress'].astype(int)

pickle_path = "D:\\Projects\\Mini_Projects\\Mental_Health_Survey\\Model\\cleaning.pkl"
with open(pickle_path, "wb") as f:
    pickle.dump(data, f)
