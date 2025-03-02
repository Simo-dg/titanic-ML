file_path = '/content/drive/My Drive/titanic.csv'

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV

df = pd.read_csv(file_path)
df.head()

df['FamilySize'] = df['SibSp'] + df['Parch']
df['IsAlone'] = (df['FamilySize'] == 0).astype(int)
df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 60, 100], labels=['Child', 'Adult', 'Senior'])

y = df['Survived']
X = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone', 'Title', 'AgeGroup', 'Sex']]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

numeric_features = ['Pclass',  'Fare', 'FamilySize']
categorical_features = ['Embarked', 'Sex', 'IsAlone', 'Title', 'AgeGroup']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean'))
        ]), numeric_features),
        
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),  
            ('encoder', OneHotEncoder(handle_unknown='ignore'))  
        ]), categorical_features)
    ]
)

my_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier()) 
])


my_pipeline.fit(X_train, y_train)
y_pred = my_pipeline.predict(X_test)

score = classification_report(y_test, y_pred)
print(score)

#now try to optimise hyperparameters
param_dist = {
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__n_estimators': [100, 150, 200],
    'classifier__max_depth': [3, 4, 5],
    'classifier__min_child_weight': [1, 2],
    'classifier__subsample': [0.7, 0.8, 1.0],
    'classifier__colsample_bytree': [0.7, 0.8, 1.0],
    'classifier__gamma': [0, 0.1],
    'classifier__scale_pos_weight': [1, 2, 5] 
}

random_search = RandomizedSearchCV(
    my_pipeline, 
    param_distributions=param_dist, 
    n_iter=10, 
    scoring='roc_auc',  
    cv=5, 
    verbose=1, 
    random_state=42, 
    n_jobs=-1
)


random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred))
