from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess data
carDfPath = 'used_car2.csv'
carDf = pd.read_csv(carDfPath)

features = ['car name', 'year', 'mileage (km)', 'transmission', 'rear camera', 'sun roof', 'auto retract mirror',
            'electric parking brake', 'map navigator', 'vehicle stability control', 'keyless push start',
            'sports mode', '360 camera view', 'power sliding door', 'auto cruise control']
target = 'price (Rp)'

# Handle outliers using IQR method
Q1 = carDf[target].quantile(0.25)
Q3 = carDf[target].quantile(0.75)
IQR = Q3 - Q1
carDf_cleaned = carDf[~((carDf[target] < (Q1 - 1.5 * IQR)) | (carDf[target] > (Q3 + 1.5 * IQR)))]

# Split data
X_cleaned = carDf_cleaned[features]
y_cleaned = carDf_cleaned[target]
X_train_cleaned, X_test_cleaned, y_train_cleaned, y_test_cleaned = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

# Preprocessing pipelines
numerical_features = ['year', 'mileage (km)']
categorical_features = ['car name', 'transmission']
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numerical_features),
                                               ('cat', categorical_transformer, categorical_features)])

# Define and train the initial Random Forest model
model_rf = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', RandomForestRegressor(random_state=42))])

param_grid_rf = {'regressor__n_estimators': [100, 200],
                 'regressor__max_features': ['auto', 'sqrt', 'log2']}

grid_search_rf = GridSearchCV(model_rf, param_grid_rf, cv=5)
grid_search_rf.fit(X_train_cleaned, y_train_cleaned)
best_model_rf = grid_search_rf.best_estimator_

# Define and train the K-Nearest Neighbors model
model_knn = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', KNeighborsRegressor())])
model_knn.fit(X_train_cleaned, y_train_cleaned)

# Extract unique values for Car Name and Transmission Type
car_names = carDf['car name'].unique()
transmission_types = carDf['transmission'].unique()

# Flask routes
@app.route('/')
def index():
    return render_template('index.html', car_names=car_names, transmission_types=transmission_types)

@app.route('/predict', methods=['POST'])
def predict():
    car_name = request.form['car_name']
    year = int(request.form['year'])
    mileage = int(request.form['mileage'])
    transmission = request.form['transmission']
    
    new_data = pd.DataFrame({
        'car name': [car_name],
        'year': [year],
        'mileage (km)': [mileage],
        'transmission': [transmission],
        'rear camera': [0],
        'sun roof': [0],
        'auto retract mirror': [0],
        'electric parking brake': [0],
        'map navigator': [0],
        'vehicle stability control': [0],
        'keyless push start': [0],
        'sports mode': [0],
        '360 camera view': [0],
        'power sliding door': [0],
        'auto cruise control': [0]
    })

    predicted_price_rf = best_model_rf.predict(new_data)[0]
    predicted_price_knn = model_knn.predict(new_data)[0]

    return render_template('index.html', 
                           car_names=car_names, 
                           transmission_types=transmission_types,
                           predicted_price_rf=predicted_price_rf, 
                           predicted_price_knn=predicted_price_knn,
                           car_name=car_name,
                           year=year,
                           mileage=mileage,
                           transmission=transmission)

if __name__ == "__main__":
    app.run(debug=True)
