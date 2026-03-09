'''Implement a Model for the titanic data'''
import pandas as pd
import numpy as np
import os , joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier , VotingClassifier
from sklearn.metrics import accuracy_score , precision_score , mean_squared_error
import xgboost as xgb

# Read the dataset using a robust path
current_dir = os.path.dirname(os.path.abspath(__file__))
train_data_dir = os.path.join(current_dir , '..', 'data', 'dataset.csv')
train_data = pd.read_csv(train_data_dir)

#Drop unnecessary columns
train_data = train_data.drop(columns=['PassengerId' , 'Name' , 'Ticket' , 'Cabin'])

# Reduce the features by adding a family sizes column 
train_data['fsize'] = train_data['SibSp'] + train_data['Parch'] #Add siblings , spouses , parents and children columns

# Get the X and Y data in both train and test
X = train_data.drop(columns=['Survived' , 'SibSp' , 'Parch'])
y = train_data['Survived']

X_train , X_test , y_train , y_test = train_test_split(X ,y ,test_size=0.2, random_state=42)

# Get numerical and categorical feature data types columns
numerical_selector = make_column_selector(dtype_exclude=object)
categorical_selector = make_column_selector(dtype_include=object)

numerical_features = numerical_selector(X)
categorical_features = categorical_selector(X)

# Create a pipeline for each feature
numerical_pipeline = Pipeline(steps=[
    ('imputer' , SimpleImputer(missing_values=np.nan ,strategy='most_frequent')) ,
    ('normalizer' , StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer' , SimpleImputer(strategy='most_frequent')), 
    ('encoder' , OneHotEncoder(handle_unknown='ignore'))
])

#Create a transfomer for the pipelines and features
column_transformer = ColumnTransformer(
    transformers=[
        ('num' , numerical_pipeline , numerical_features) ,
        ('cat' , categorical_pipeline , categorical_features)
    ]
)

#Create a pipeline for the transformer and the classifier model
classifier = VotingClassifier(estimators=[
    ('forest' , RandomForestClassifier()) , 
    ('xgb' , xgb.XGBClassifier())
])

# classifier = RandomForestClassifier()

model_pipeline = Pipeline(steps=[
    ('transformer' , column_transformer) ,
    ('classifier' , classifier)
])

model_pipeline.fit(X=X_train ,y=y_train)

y_pred = model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test , y_pred)
precision = precision_score(y_test , y_pred)
mse = mean_squared_error(y_test , y_pred)

print(f'Accuracy : {accuracy:.4f}')
print(f'Precision : {precision:.4f}')
print(f'MSE : {mse:.4f}')

def save_model():
    print('Saving Pipeline ...')
    model_save_path = os.path.join(current_dir, '..', 'models', 'model_pipeline.pkl')
    joblib.dump(model_pipeline , model_save_path)

save_model()


