import pandas as pd
pd.options.mode.chained_assignment = None 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import json

train_data = pd.read_json('C:/Users/Marius/Desktop/SII/pred/train.json')
test_data = pd.read_json('C:/Users/Marius/Desktop/SII/pred/test.json')

'''missing_values = train_data.isnull().sum()
print("Numărul total de valori lipsă în fiecare coloană:")
print(missing_values)'''

train_data['addons'] = train_data['addons'].apply(lambda x: ', '.join(x))
test_data['addons'] = test_data['addons'].apply(lambda x: ', '.join(x))

X_train, X_test, y_train, y_test = train_test_split(
    train_data.drop('pret', axis=1),
    train_data['pret'],
    test_size=0.2,
    random_state=42
)

numeric_features = ['an', 'km', 'putere','capacitate_cilindrica']
text_features = ['marca', 'model', 'cutie_de_viteze','combustibil','transmisie','caroserie','culoare','optiuni_culoare','addons']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
])

text_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='')), 
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('text', text_transformer, text_features)
    ]
)

model = RandomForestRegressor(n_estimators=100, random_state=42)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")

predictions = pipeline.predict(test_data.drop('pret', axis=1))

result_list = []
for index, row in test_data.iterrows():
    result_dict = {
        "marca": row["marca"],
        "model": row["model"],
        "an": row["an"],
        "km": row["km"],
        "putere": row["putere"],
        "cutie_de_viteze": row["cutie_de_viteze"],
        "combustibil": row["combustibil"],
        "capacitate_cilindrica": row["capacitate_cilindrica"],
        "transmisie": row["transmisie"],
        "caroserie": row["caroserie"],
        "culoare":row["culoare"],
        "optiuni_culoare": row["optiuni_culoare"],
        "addons": row["addons"],
        "pret": predictions[index], 
    }
    result_list.append(result_dict)

output_json_file = 'C:/Users/Marius/Desktop/SII/pred/test.json'
with open(output_json_file, 'w', encoding='utf-8') as json_file:
    json.dump(result_list, json_file, indent=4)

