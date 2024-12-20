import numpy as np
import pandas as pd
import shap

from NN_callbacks import StopAtMinLR, AdjustBatchSizeOnLR, StopAtLossValue
from tensorflow.keras.callbacks import ReduceLROnPlateau
from models import get_NN_model, Ensemble, AveragingEnsemble
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error,mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def predict_model(df_train, df_test, model_config):

    y = df_train['SalePrice'].to_numpy()
    X = df_train.drop(columns=["SalePrice"]).to_numpy()
    X_test = df_test.to_numpy()
    
    log_predictions = create_predictions(X, y, X_test, model_config, )
    predictions = np.exp(log_predictions)
            
    return predictions

def predict_ensemble(df_train, df_test, model_configs):

    y = df_train['SalePrice'].to_numpy()
    X = df_train.drop(columns=["SalePrice"]).to_numpy()
    X_test = df_test.to_numpy()
    
    ensemble_model = Ensemble(model_configs)
    
    ensemble_model.fit(X, y)
    ensemble_log_predictions = ensemble_model.predict(X_test)
    predictions = np.exp(ensemble_log_predictions)

    return predictions

def predict_GBRT_k_fold(df, df_test, model_config, K):

    y = df['SalePrice'].to_numpy()
    X = df.drop(columns=["SalePrice"]).to_numpy()
    X_test = df_test.to_numpy()
    K_estimators = []
    random_state =42

    kf = KFold(n_splits=K, shuffle=True, random_state = random_state)

    fold = 0

    for train_index, val_index in kf.split(X):
        print(f"Processing fold #{fold + 1} of {K}")

        # Split data into training and validation sets
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        gbrt_clf = GradientBoostingRegressor(max_depth = model_config.max_depth, n_estimators = model_config.n_estimators, learning_rate = model_config.learning_rate)
        gbrt_clf.fit(X_train, y_train)

        errors = [mean_squared_error(y_val, y_pred) for y_pred in gbrt_clf.staged_predict(X_val)]
        bst_n_estimators = np.argmin(errors)
        print(f'best_n_estinators is: {bst_n_estimators}')

        K_estimators.append(bst_n_estimators)

    min_n_estimator = np.min(K_estimators)
    print(f'The n_estinators used in the final model is: {min_n_estimator}')

    final_gbrt = GradientBoostingRegressor(
    max_depth=model_config.max_depth,
    n_estimators=min_n_estimator,
    learning_rate=model_config.learning_rate
    )

    final_gbrt.fit(X, y)

    log_predicts = final_gbrt.predict(X_test)
    predictions = np.exp(log_predicts)

    return predictions

def create_predictions(X_train, y_train, X_test, model_config):
                
    if model_config.model_name == 'gbrt':
        print('Start gbrt training')

        gbrt = GradientBoostingRegressor(n_estimators=model_config.n_estimators, learning_rate=model_config.learning_rate, max_depth=model_config.max_depth, random_state=42)
        gbrt.fit(X_train, y_train)

        log_predicts = gbrt.predict(X_test).reshape(-1, 1)

    elif model_config.model_name == "neural_network":
        print('Start neural network training')

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
                            callbacks=callbacks, verbose=1)
        
        log_predicts = gbrt.predict(X_test).reshape(-1, 1)
    
    elif model_config.model_name == 'random_forest':
        print('Start random forest regression training')

        rfr = RandomForestRegressor(n_estimators = model_config.n_estimators, max_leaf_nodes = model_config.max_leaf_nodes, n_jobs = model_config.n_jobs)
        rfr.fit(X_train, y_train)

        log_predicts = gbrt.predict(X_test).reshape(-1, 1)
        
    return log_predicts

def predict_shap(df_train, df_test, model_config):

    y = df_train['SalePrice'].to_numpy()
    X = df_train.drop(columns=["SalePrice"]).to_numpy()
    X_test = df_test.to_numpy()
    
    if model_config.model_name == 'gbrt':
        print('Start gbrt training')

        gbrt = GradientBoostingRegressor(n_estimators=model_config.n_estimators, learning_rate=model_config.learning_rate, max_depth=model_config.max_depth, random_state=42)
        gbrt.fit(X, y)

        explainer = shap.Explainer(gbrt, X)

        # Explain predictions for the test dataset
        shap_values = explainer(X_test)
        print(type(shap_values))

        # Aggregate SHAP values by test rows
        shap.summary_plot(shap_values, X_test)

    return shap_values

def predict_avg(df_train, df_test, model_configs, use_specfic_state):

    y = df_train['SalePrice']#.to_numpy()
    X = df_train.drop(columns=["SalePrice"])#.to_numpy()
    X_test = df_test.to_numpy()
    
    ensemble_model = AveragingEnsemble(model_configs)
    
    log_predict_train = ensemble_model.fit(X, y, use_specfic_state)
    log_predictions_test = ensemble_model.predict(df_test)

    predictions_test = pd.DataFrame()

    predictions_test['avg_predictions'] = np.exp(log_predictions_test.mean(axis=1))

    return predictions_test
