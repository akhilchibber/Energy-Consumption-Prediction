import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm 
import datetime
from meteocalc import feels_like, Temp
import gc
import os
import argparse

categorical = ['building_id', 'site_id', 'primary_use', 'meter', 'dayofweek']
features = [] 

from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

def reduce_mem_usage(df, use_float16=False):
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
    
    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]): #is_categorical_dtype is deprecated
            continue
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    return df

def fill_weather_dataset(weather_df):
    time_format = "%Y-%m-%d %H:%M:%S"
    # Handle cases where weather_df might be empty or have no valid timestamps
    if weather_df.empty or weather_df['timestamp'].isnull().all():
        print("Weather data is empty or has no valid timestamps for min/max in fill_weather_dataset.")
        # Return a DataFrame with expected columns but no data, or handle as error
        # For now, let's ensure it has the columns to avoid later errors if it's used.
        cols = ['site_id', 'timestamp', 'air_temperature', 'cloud_coverage', 'dew_temperature', 
                'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed',
                'relative_humidity', 'feels_like'] # Expected columns after processing
        empty_weather = pd.DataFrame(columns=cols)
        # Convert timestamp to datetime if creating an empty df with schema
        if 'timestamp' in empty_weather.columns:
             empty_weather['timestamp'] = pd.to_datetime(empty_weather['timestamp'])
        return empty_weather


    start_date = datetime.datetime.strptime(weather_df['timestamp'].min(),time_format)
    end_date = datetime.datetime.strptime(weather_df['timestamp'].max(),time_format)
    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)
    hours_list = [(end_date - datetime.timedelta(hours=x)).strftime(time_format) for x in range(total_hours)]

    for site_id in range(16): # Assuming 16 sites as per original notebook
        site_hours = np.array(weather_df[weather_df['site_id'] == site_id]['timestamp'])
        new_rows = pd.DataFrame(np.setdiff1d(hours_list,site_hours),columns=['timestamp'])
        new_rows['site_id'] = site_id
        weather_df = pd.concat([weather_df,new_rows], ignore_index=True) 

    weather_df = weather_df.reset_index(drop=True)           

    weather_df["datetime"] = pd.to_datetime(weather_df["timestamp"])
    weather_df["day"] = weather_df["datetime"].dt.day
    weather_df["week"] = weather_df["datetime"].dt.isocalendar().week.astype(int)
    weather_df["month"] = weather_df["datetime"].dt.month
    
    weather_df = weather_df.set_index(['site_id','day','month'])

    for col in ['air_temperature', 'cloud_coverage', 'dew_temperature', 'sea_level_pressure', 'wind_direction', 'wind_speed', 'precip_depth_1_hr']:
        filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])[col].mean(), columns=[col])
        if col in ['cloud_coverage', 'sea_level_pressure', 'precip_depth_1_hr']: # These had ffill in notebook
             filler[col] = filler[col].fillna(method='ffill')
        weather_df.update(filler,overwrite=False)
    
    weather_df = weather_df.reset_index()
    weather_df = weather_df.drop(['datetime','day','week','month'],axis=1)
    
    def get_meteorological_features(data):
        if not data.empty and 'dew_temperature' in data.columns and 'air_temperature' in data.columns:
            data['relative_humidity'] = 100 * (np.exp((17.625 * data['dew_temperature']) / (243.04 + data['dew_temperature'])) / np.exp((17.625 * data['air_temperature'])/(243.04 + data['air_temperature'])))
        else:
            data['relative_humidity'] = np.nan

        if not data.empty and 'air_temperature' in data.columns and 'relative_humidity' in data.columns and 'wind_speed' in data.columns:
            flike_final = []
            for i in range(len(data)):
                at = data['air_temperature'].iloc[i] 
                rh = data['relative_humidity'].iloc[i]
                ws = data['wind_speed'].iloc[i]
                if pd.notnull(at) and pd.notnull(rh) and pd.notnull(ws) and ws >= 0:
                    flike_final.append(feels_like(Temp(at, unit = 'C'), rh, ws).f)
                else:
                    flike_final.append(np.nan) 
            data['feels_like'] = flike_final
        else:
            data['feels_like'] = np.nan
        return data

    weather_df = get_meteorological_features(weather_df)
    return weather_df

def features_engineering(df):
    df.sort_values("timestamp", inplace=True) 
    df.reset_index(drop=True, inplace=True) 
    
    df["timestamp"] = pd.to_datetime(df["timestamp"],format="%Y-%m-%d %H:%M:%S")
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    
    df['month_group'] = df['timestamp'].dt.month 
    df['month_group'].replace((1, 2, 3, 4), 1, inplace = True)
    df['month_group'].replace((5, 6, 7, 8), 2, inplace = True)
    df['month_group'].replace((9, 10, 11, 12), 3, inplace = True)
  
    if 'square_feet' in df.columns:
        df['square_feet'] =  np.log1p(df['square_feet'])
    
    drop_cols = ["timestamp"] 
    df = df.drop(drop_cols, axis=1, errors='ignore') # Added errors='ignore'
    gc.collect()
    
    if 'primary_use' in df.columns:
        le = LabelEncoder()
        df["primary_use"] = le.fit_transform(df["primary_use"])
    
    return df

class DummyModel:
    def predict(self, data):
        print(f"DummyModel: Predicting with {len(data)} rows of data.")
        return np.full(len(data), np.nan) # Return NaNs for dummy predictions

def generate_predictions(test_csv_path, building_metadata_csv_path, weather_test_csv_path, output_csv_path, model_dir="."):
    global features 

    print("Loading data...")
    test_df = pd.read_csv(test_csv_path)
    row_ids = test_df["row_id"] 
    test_df.drop("row_id", axis=1, inplace=True)
    test_df = reduce_mem_usage(test_df)

    building_df = pd.read_csv(building_metadata_csv_path)
    building_df = reduce_mem_usage(building_df, use_float16=True)
    test_df = test_df.merge(building_df,left_on='building_id',right_on='building_id',how='left')
    del building_df
    gc.collect()

    weather_df = pd.read_csv(weather_test_csv_path)
    weather_df = fill_weather_dataset(weather_df) # fill_weather_dataset now handles empty/min timestamp issues
    weather_df = reduce_mem_usage(weather_df)
    
    test_df['timestamp_merge_key'] = pd.to_datetime(test_df["timestamp"], format="%Y-%m-%d %H:%M:%S")
    weather_df['timestamp_merge_key'] = pd.to_datetime(weather_df["timestamp"], format="%Y-%m-%d %H:%M:%S", errors='coerce') # Coerce errors for safety

    test_df = test_df.merge(weather_df,how='left',on=['site_id','timestamp_merge_key'], suffixes=('', '_weather'))
    test_df.drop('timestamp_merge_key', axis=1, inplace=True, errors='ignore')
    if 'timestamp_weather' in test_df.columns: test_df.drop('timestamp_weather', axis=1, inplace=True)
    if 'timestamp' not in test_df.columns and 'timestamp_x' in test_df.columns: 
        test_df.rename(columns={'timestamp_x':'timestamp'}, inplace=True)
    elif 'timestamp_y' in test_df.columns: 
        test_df.drop('timestamp_y', axis=1, inplace=True, errors='ignore')

    del weather_df
    gc.collect()

    print("Performing feature engineering...")
    test_df = features_engineering(test_df)
    
    # Define features based on expected columns after preprocessing, matching training
    # This list must be stable and derived from the notebook's feature engineering logic.
    # Example columns based on notebook (ensure these are all generated for test_df):
    base_cols = ['building_id', 'site_id', 'primary_use', 'meter', 'dayofweek', 'square_feet', 'hour']
    weather_cols = ['air_temperature', 'dew_temperature', 'cloud_coverage', 'wind_direction', 
                    'wind_speed', 'sea_level_pressure', 'precip_depth_1_hr', 
                    'relative_humidity', 'feels_like'] 
    # Add year_built, floor_count if they are used and exist after merge
    if 'year_built' in test_df.columns: base_cols.append('year_built')
    if 'floor_count' in test_df.columns: base_cols.append('floor_count')
        
    features = [col for col in base_cols + weather_cols if col in test_df.columns]
    print(f"Using features: {features}")

    models = []
    num_folds = 5
    print(f"Loading {num_folds} models...")
    DUMMY_MODEL_MODE = os.environ.get("DUMMY_MODEL_MODE") == "true"

    for fold_idx in range(num_folds):
        model_path = os.path.join(model_dir, f'lgbm_model_fold_{fold_idx}.txt')
        if not os.path.exists(model_path):
            if DUMMY_MODEL_MODE:
                print(f"DUMMY_MODEL_MODE: Model file {model_path} not found. Creating dummy model object.")
                models.append(DummyModel())
                continue
            else:
                print(f"Model file {model_path} not found. Ensure models are in the specified directory.")
                return 
        
        try:
            model = lgb.Booster(model_file=model_path) 
            models.append(model)
        except lgb.basic.LightGBMError as e:
            if DUMMY_MODEL_MODE:
                print(f"DUMMY_MODEL_MODE: Failed to load model {model_path} ({e}). Creating dummy model object.")
                models.append(DummyModel())
            else:
                print(f"Error loading model {model_path}. If testing with dummy model files, set DUMMY_MODEL_MODE=true environment variable.")
                raise e
    print("Models loaded.")
    
    set_size = len(test_df)
    iterations = 120 
    if set_size == 0:
        print("Test data is empty after processing. Skipping predictions.")
        # Create an empty submission file with headers if row_ids is also empty or handle as error
        if not row_ids.empty:
            submission_df = pd.DataFrame({'row_id': row_ids, 'meter_reading': []}) # Ensure meter_reading matches length
        else: # Handle case where row_ids might be empty
             submission_df = pd.DataFrame(columns=['row_id', 'meter_reading'])
        submission_df.to_csv(output_csv_path, index=False)
        print(f"Empty submission file created: {output_csv_path}")
        return

    batch_size = (set_size + iterations - 1) // iterations 
    meter_reading = []
    print(f"Starting predictions for {set_size} rows in {iterations} iterations (batch size ~{batch_size})...")
    
    for i in tqdm(range(iterations)):
        pos = i*batch_size
        end_pos = min(pos + batch_size, set_size)
        if pos >= end_pos:
            continue 
        
        current_batch_df = test_df[features].iloc[pos : end_pos]
        # Ensure all features are present in current_batch_df, fill with NaN if not (e.g. if some features were conditional)
        for feature_col in features:
            if feature_col not in current_batch_df.columns:
                current_batch_df[feature_col] = np.nan
        
        fold_preds = [np.expm1(model.predict(current_batch_df[features])) for model in models]
        meter_reading.extend(np.mean(fold_preds, axis=0))

    print(f"Total predictions generated: {len(meter_reading)}")
    if len(meter_reading) != set_size:
        print(f"Warning: Prediction length mismatch. Expected {set_size}, got {len(meter_reading)}. Padding with NaNs.")
        meter_reading.extend([np.nan] * (set_size - len(meter_reading)))
    
    submission_df = pd.DataFrame({'row_id': row_ids, 'meter_reading': meter_reading[:len(row_ids)]})
    submission_df['meter_reading'] = np.clip(submission_df['meter_reading'], a_min=0, a_max=None)
    submission_df.to_csv(output_csv_path, index=False)
    print(f"Submission file created: {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate energy consumption predictions.")
    parser.add_argument("test_csv", help="Path to the test.csv file")
    parser.add_argument("building_metadata_csv", help="Path to the building_metadata.csv file")
    parser.add_argument("weather_test_csv", help="Path to the weather_test.csv file")
    parser.add_argument("output_csv", help="Path to save the submission.csv file")
    parser.add_argument("--model_dir", default=".", help="Directory containing the lgbm_model_fold_*.txt files (default: current directory)")
    
    args = parser.parse_args()
    
    # For testing purposes, to simulate the DUMMY_MODEL_MODE for Part C/D if needed
    # os.environ["DUMMY_MODEL_MODE"] = "true" 
    
    generate_predictions(args.test_csv, args.building_metadata_csv, args.weather_test_csv, args.output_csv, args.model_dir)
    print(f"Predictions successfully saved to {args.output_csv}")
