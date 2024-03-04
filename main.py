# flask, scikit-learn, pandas, pickle-mixin

import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np

data = pd.read_csv('Cleaned_data.csv')
app = Flask(__name__)
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))
def prepare_input_data(user_input):
    # Preprocess the user input data
    user_input_df = pd.DataFrame(user_input, columns=['location', 'total_sqft', 'bath', 'bhk'])

    # Ensure the 'price' column is not present (exclude it from input features)
    if 'price' in user_input_df.columns:
        user_input_df = user_input_df.drop(columns=['price'])

    # Convert 'total_sqft' to numeric
    def convertRange(x):
        if isinstance(x, str):
            temp = x.split('_')
            if len(temp) == 2:
                return (float(temp[0]) + float(temp[1])) / 2
            try:
                return float(x)
            except:
                return None
        else:
            return float(x)

    user_input_df['total_sqft'] = user_input_df['total_sqft'].apply(convertRange)

    # Clean up 'location' column
    user_input_df['location'] = user_input_df['location'].apply(lambda x: x.strip())

    # One-hot encode the 'location' column
    user_input_encoded = pd.get_dummies(user_input_df, columns=['location'])

    # Ensure all columns from the training data are present in the input data
    missing_cols = set(pipe.named_steps['columntransformer'].get_feature_names_out()) - set(user_input_encoded.columns)
    for col in missing_cols:
        user_input_encoded[col] = 0  # Add missing columns and set them to 0

    return user_input_encoded

def preprocess_data(df):
    # Handle missing values
    df['location'] = df['location'].fillna('sarjapur Road')
    df['bath'] = df['bath'].fillna(df['bath'].median())


    # Clean up 'location' column
    df['location'] = df['location'].apply(lambda x: x.strip())
    location_count = df['location'].value_counts()
    location_count_less_10 = location_count[location_count <= 10]
    df['location'] = df['location'].apply(lambda x: 'other' if x in location_count_less_10 else x)

    # Check if the columns exist before dropping
    columns_to_drop = ['size', 'price_per_sqft', 'price']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

    # Calculate 'price_per_sqft'
    df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']

    # Remove outliers
    def remove_outliers_sqft(df):
        df_output = pd.DataFrame()
        for key, subdf in df.groupby('location'):
            m = np.mean(subdf.price_per_sqft)
            st = np.std(subdf.price_per_sqft)
            gen_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
            df_output = pd.concat([df_output, gen_df], ignore_index=True)
        return df_output

    df = remove_outliers_sqft(df)

    # Remove outliers based on 'bhk'
    def bhk_outlier_remover(df):
        exclude_indices = np.array([])
        for location, location_df in df.groupby('location'):
            bhk_stats = {}
            for bhk, bhk_df in location_df.groupby('bhk'):
                bhk_stats[bhk] = {
                    'mean': np.mean(bhk_df.price_per_sqft),
                    'std': np.std(bhk_df.price_per_sqft),
                    'count': bhk_df.shape[0]
                }
            print(location, bhk_stats)
            for bhk, bhk_df in location_df.groupby('bhk'):
                stats = bhk_stats.get(bhk - 1)
                if stats and stats['count'] > 5:
                    exclude_indices = np.append(exclude_indices,
                                                bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
        return df.drop(exclude_indices, axis='index')

    df = bhk_outlier_remover(df)

    # Drop unnecessary columns
    df.drop(columns=['size', 'price_per_sqft'], inplace=True)

    return df


# ... (your existing code)



@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    locations = request.form.get('location')
    bhk = float(request.form.get('bhk'))
    bath = float(request.form.get('bath'))
    sqft = float(request.form.get('total_sqft'))

    # Prepare input data for prediction
    input_data = [[locations, bhk, bath, sqft]]
    user_input_encoded = prepare_input_data(input_data)

    # Now, you can use user_input_encoded for prediction
    prediction = pipe.predict(user_input_encoded)[0]

    return str(np.round(prediction))


if __name__ == '__main__':
    app.run(debug=True, port=5001)