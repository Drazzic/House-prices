import numpy as np
import pandas as pd
import shap
import tensorflow as tf
import random

import matplotlib.pyplot as plt
from NN_callbacks import StopAtMinLR, AdjustBatchSizeOnLR, StopAtLossValue
from tensorflow.keras.callbacks import ReduceLROnPlateau
from models import get_NN_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error,mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from catboost import CatBoostRegressor

from data_transformers import ColumnDescriptor

def compute_fold_averages(scores, K, I):
    """Compute averages across iterations and folds."""
    lowest_epochs = min(len(sublist) for iter_scores in scores for sublist in iter_scores)
    flattened_scores = [item[:lowest_epochs] for iter_scores in scores for item in iter_scores]
    return [np.mean([x[i] for x in flattened_scores]) for i in range(lowest_epochs)]

def plot_graph(train_scores, val_scores, label, start_epoch=0):
    """Plot training and validation scores."""
    epochs = range(1 + start_epoch, len(train_scores) + 1 + start_epoch)
    plt.plot(epochs, train_scores, 'r', label=f"{label} (Train)")
    plt.plot(epochs, val_scores, 'b', label=f"{label} (Validation)")
    plt.xlabel("Epochs")
    plt.ylabel(label)
    plt.legend()
    plt.show()


def evaluate_model(df, K, model_config):

    y = df['SalePrice']

    X = df.drop(columns=["SalePrice"])
    print(X.shape)

    if K > 1:
    
        kf = KFold(n_splits=K, shuffle=True, random_state = model_config.random_state)

        fold_rmse_train = []
        fold_rmse_val = []

        fold = 0

        for train_index, val_index in kf.split(X):
            print(f"Processing fold #{fold + 1} of {K}")

            # Split data into training and validation sets
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                    
            train_preds, val_preds = create_predictions(X_train, y_train, X_val, y_val, model_config)
            rmse_train = root_mean_squared_error(y_train, train_preds)
            rmse_val = root_mean_squared_error(y_val, val_preds)    
            print(f"{model_config.model_name}, Fold #{fold + 1} - RMSE (Train): {rmse_train:.6f}, RMSE (Validation): {rmse_val:.6f}")

            fold_rmse_train.append(rmse_train)
            fold_rmse_val.append(rmse_val)
            fold += 1

        return fold_rmse_train, fold_rmse_val

    else:

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=model_config.random_state)
            
        train_preds, val_preds = create_predictions(X_train, y_train, X_val, y_val, model_config)

        rmse_train = root_mean_squared_error(y_train, train_preds)
        rmse_val = root_mean_squared_error(y_val, val_preds)

        return rmse_train, rmse_val

def evaluate_ensemble(df, K, model_configs):
    
    nmbr_of_models = len(model_configs)
    
    y = df['SalePrice']
    X = df.drop(columns=["SalePrice"])

    if K > 1:

        fold_rmse_train = []
        fold_rmse_val = []

        kf = KFold(n_splits=K, shuffle=True, random_state=42)

        fold = 0
        for train_index, val_index in kf.split(X):

            # Split data into training and validation sets
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            print(f"Processing fold #{fold + 1} of {K}")

            for model_nmbr in range(nmbr_of_models):

                model_config = model_configs[model_nmbr]
            
                train_preds, val_preds = create_predictions(X_train, y_train, X_val, y_val, model_config)
                
                X_train = np.hstack((X_train, train_preds))
                X_val = np.hstack((X_val, val_preds))

            rmse_train = root_mean_squared_error(y_train, train_preds)
            rmse_val = root_mean_squared_error(y_val, val_preds)
            print(f"Ensemble, Fold #{fold + 1} - RMSE (Train): {rmse_train:.4f}, RMSE (Validation): {rmse_val:.4f}")

            fold_rmse_train.append(rmse_train)
            fold_rmse_val.append(rmse_val)
            fold += 1
        
        return fold_rmse_train, fold_rmse_val

    else:
            
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        for model_nmbr in range(nmbr_of_models):

            model_config = model_configs[model_nmbr]

            train_preds, val_preds = create_predictions(X_train, y_train, X_val, y_val, model_config)

            X_train = np.hstack((X_train, train_preds))
            X_val = np.hstack((X_val, val_preds))
        
        rmse_train = root_mean_squared_error(y_train, train_preds)
        rmse_val = root_mean_squared_error(y_val, val_preds)
        print(f"Ensemble, RMSE (Train): {rmse_train:.4f}, RMSE (Validation): {rmse_val:.4f}")

        return rmse_train, rmse_val
    
def evaluate_shap(df, model_config):

    y = df['SalePrice']#.to_numpy()

    X = df.drop(columns=["SalePrice"])#.to_numpy()

    gbrt = GradientBoostingRegressor(n_estimators=model_config.n_estimators, learning_rate=model_config.learning_rate, max_depth=model_config.max_depth, random_state=42)
    gbrt.fit(X, y)

    predictions = gbrt.predict(X)#.reshape(-1, 1)
    print(predictions)

    # Assuming y is your true target values and predictions is your model's output
    errors = (y - predictions.flatten()) / y  # Calculate signed errors

    # Create a DataFrame to organize and sort
    results = pd.DataFrame({
        'Actual': y,
        'Predicted': predictions.flatten(),
        'Error': errors
    })

    # Find the 10 worst too-high predictions (negative errors sorted ascending)
    too_high = results[results['Error'] < 0].sort_values(by='Error').head(10)

    # Find the 10 worst too-low predictions (positive errors sorted descending)
    too_low = results[results['Error'] > 0].sort_values(by='Error', ascending=False).head(10)

    X_too_high = X.iloc[too_high.index]
    X_too_low = X.iloc[too_low.index]

    print("10 Worst Too-High Predictions:")
    print(too_high)

    print("\n10 Worst Too-Low Predictions:")
    print(too_low)

    explainer = shap.Explainer(gbrt, X)

    def analyze_single_prediction(row_index, X_subset, subset_type="Too High", original_index=None, error_value=None):
        print(f"SHAP Analysis for {subset_type} Prediction at Rank {row_index + 1}")
        single_row = X_subset[row_index:row_index + 1]  # Keep it as a 2D array
        shap_values_single = explainer(single_row)  # Compute SHAP values
        
        custom_title = f"{subset_type} Prediction at Index {original_index}, Error: {error_value:.4f}"
        shap.waterfall_plot(shap_values_single[0], max_display=20, show=False)

        # Add a custom title
        
        plt.title(custom_title)
        plt.show()

    # Explain predictions for the test dataset
    # Loop through too-high predictions
    for i, (idx, row) in enumerate(too_high.iterrows()):
        analyze_single_prediction(
            row_index=i,
            X_subset=X_too_high,
            subset_type="Too High",
            original_index=idx,
            error_value=row["Error"]
        )

    # Loop through too-low predictions
    for i, (idx, row) in enumerate(too_low.iterrows()):
        analyze_single_prediction(
            row_index=i,
            X_subset=X_too_low,
            subset_type="Too Low",
            original_index=idx,
            error_value=row["Error"]
        )

    return predictions


def create_predictions(X_train, y_train, X_val, y_val, model_config):
                
    if model_config.model_name == 'gbrt':
        print('Start gbrt training')

        gbrt = GradientBoostingRegressor(n_estimators=model_config.n_estimators, learning_rate=model_config.learning_rate, max_depth=model_config.max_depth, random_state=model_config.random_state)
        gbrt.fit(X_train, y_train)

        train_preds = gbrt.predict(X_train).reshape(-1, 1)
        val_preds = gbrt.predict(X_val).reshape(-1, 1)

    elif model_config.model_name == "neural_network":
        print('Start neural network training')

        random.seed(model_config.random_state)             # Python random seed
        np.random.seed(model_config.random_state) 
        tf.random.set_seed(model_config.random_state) 

        # Build and fit the model
        nn_model = get_NN_model(model_config.metric, model_config.learning_rate)

        if model_config.callbacks:
            
            callbacks = []

            for callback_name, params in model_config.callbacks.items():
                if callback_name == "StopAtMinLR":
                    callbacks.append(StopAtMinLR(**params))
                elif callback_name == "ReduceLROnPlateau":
                    callbacks.append(ReduceLROnPlateau(**params))
                elif callback_name == "AdjustBatchSizeOnLR":
                    callbacks.append(AdjustBatchSizeOnLR(**params))
                else:
                    raise ValueError(f"Unknown callback: {callback_name}")
        
        nn_model.fit(X_train, y_train, epochs=model_config.epochs, batch_size=model_config.batch_size,
                            validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
        
        train_preds = nn_model.predict(X_train).reshape(-1, 1)
        val_preds = nn_model.predict(X_val).reshape(-1, 1)
    
    elif model_config.model_name == 'random_forest':
        print('Start random forest regression training')

        rfr = RandomForestRegressor(n_estimators = model_config.n_estimators, max_leaf_nodes = model_config.max_leaf_nodes, n_jobs = model_config.n_jobs, random_state = model_config.random_state)
        rfr.fit(X_train, y_train)

        train_preds = rfr.predict(X_train).reshape(-1, 1)
        val_preds = rfr.predict(X_val).reshape(-1, 1)

    elif model_config.model_name == 'lightgbm':
        print('Start lightgbm training')

        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = {
                'metric': model_config.metric,
                'num_leaves': model_config.num_leaves,
                'learning_rate': model_config.learning_rate,
                'feature_fraction': model_config.feature_fraction,
                'num_iterations': model_config.num_iterations,
                'max_depth': model_config.max_depth,
                'early_stopping_rounds': model_config.early_stopping_rounds,
                'seed':model_config.random_state,
                'verbose': -1
                }

        model = lgb.train(train_set = train_data, valid_sets=[test_data], params=params)
                          
        train_preds = model.predict(X_train).reshape(-1, 1)
        val_preds = model.predict(X_val).reshape(-1, 1)

    elif model_config.model_name == 'catboost':

        column_descriptor = ColumnDescriptor()
        column_descriptor.fit(X_train)
        num_features, cat_features = column_descriptor.get_output_columns()

        model = CatBoostRegressor(iterations=model_config.iterations, depth=model_config.depth, learning_rate=model_config.learning_rate, cat_features=cat_features, random_seed = model_config.random_seed)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=10)

        feature_names = X_train.columns

        feature_importances = model.get_feature_importance()

        importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
        importances_df = importances_df.sort_values(by='Importance', ascending=False)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(importances_df['Feature'], importances_df['Importance'], color='skyblue')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.title('Feature Importances')
        plt.gca().invert_yaxis()  # Invert y-axis for better readability
        plt.show()

        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)

    return train_preds, val_preds

def calc_shap(model, X):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Rank features by mean absolute SHAP value
    shap_importances = np.abs(shap_values).mean(axis=0)
    feature_ranking = X.columns[np.argsort(shap_importances)[::-1]]

    return feature_ranking

