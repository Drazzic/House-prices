import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error,mean_squared_error

def random_forest(df):

    K = 4

    shuffled_data = df.sample(frac=1).reset_index(drop=True)
    num_validation_samples = len(df) // K

    rmse_scores = []  # To store RMSE for each fold

    for fold in range(K):
        print(f"Processing fold #{fold + 1} of {K}")
        val_data = shuffled_data.iloc[fold * num_validation_samples: (fold + 1) * num_validation_samples]
        train_data = shuffled_data.drop(val_data.index)

        val_labels = np.log(val_data['SalePrice'])
        train_labels = np.log(train_data['SalePrice'])

        val_data = val_data.drop(columns=["SalePrice"]).to_numpy()
        train_data = train_data.drop(columns=["SalePrice"]).to_numpy()

        rnd_for_clf = RandomForestRegressor(n_estimators = 500, max_leaf_nodes = 32, n_jobs = -1)
        rnd_for_clf.fit(train_data, train_labels)

        y_pred = rnd_for_clf.predict(val_data)

        rmse = root_mean_squared_error(val_labels, y_pred)
        rmse_scores.append(rmse)
        print(f"Fold #{fold + 1} RMSE: {rmse}")

    avg_rmse = np.mean(rmse_scores)
    print(f"Average RMSE across {K} folds: {avg_rmse}")

    return avg_rmse

def GBRT(df, df_test):

    K = 4

    shuffled_data = df.sample(frac=1).reset_index(drop=True)
    num_validation_samples = len(df) // K

    rmse_scores = []  # To store RMSE for each fold
    K_estimators = []

    max_depth = 4
    learning_rate = 0.05
    for fold in range(K):
        print(f"Processing fold #{fold + 1} of {K}")
        val_data = shuffled_data.iloc[fold * num_validation_samples: (fold + 1) * num_validation_samples]
        train_data = shuffled_data.drop(val_data.index)

        val_labels = np.log(val_data['SalePrice'])
        train_labels = np.log(train_data['SalePrice'])

        val_data = val_data.drop(columns=["SalePrice"]).to_numpy()
        train_data = train_data.drop(columns=["SalePrice"]).to_numpy()

        gbrt_clf = GradientBoostingRegressor(max_depth = max_depth, n_estimators = 1000, learning_rate = learning_rate)
        gbrt_clf.fit(train_data, train_labels)

        errors = [mean_squared_error(val_labels, y_pred) for y_pred in gbrt_clf.staged_predict(val_data)]
        bst_n_estimators = np.argmin(errors)
        print(f'best_n_estinators is: {bst_n_estimators}')

        gbrt_best = GradientBoostingRegressor(max_depth=max_depth, n_estimators=bst_n_estimators, learning_rate = learning_rate)
        gbrt_best.fit(train_data, train_labels)

        y_pred = gbrt_best.predict(val_data)

        rmse = root_mean_squared_error(val_labels, y_pred)
        rmse_scores.append(rmse)
        K_estimators.append(bst_n_estimators)
        print(f"Fold #{fold + 1} RMSE: {rmse}")

    avg_rmse = np.mean(rmse_scores)
    min_n_estimator = np.min(K_estimators)
    print(f"Average RMSE across {K} folds: {avg_rmse}")

    all_train_labels = df['SalePrice']
    all_train_data = df.drop('SalePrice', axis=1)

    final_gbrt = GradientBoostingRegressor(
        max_depth=max_depth,
        n_estimators=min_n_estimator,
        learning_rate=learning_rate
    )

    final_gbrt.fit(all_train_data, all_train_labels)

    test_log_predictions = final_gbrt.predict(df_test)

    predictions = np.exp(test_log_predictions)
    pred_list = predictions.flatten().tolist()

    id_list = list(range(1461, 1461 + len(pred_list)))
    df_final = pd.DataFrame({
    'Id': id_list,
    'SalePrice': pred_list
    })

    df_final.to_csv('submission.csv', index=False)
    
    print("List saved to 'submission.csv'")

    return avg_rmse

