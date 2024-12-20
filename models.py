import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
import random
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
from NN_callbacks import StopAtMinLR, AdjustBatchSizeOnLR, StopAtLossValue
from tensorflow.keras.callbacks import ReduceLROnPlateau


class ModelConfig:
    """
    A class to define and manage model configurations.
    """
    def __init__(self, model_name, epochs=None, batch_size=None, learning_rate=None, metric= None, validation_split = None, callbacks=None, 
                 n_estimators = None, max_leaf_nodes = None, n_jobs= None, max_depth = None, depth = None, item_to_monitor = None, verbose = 1, graphs = False,
                 num_leaves = 31, feature_fraction = 0.9, num_iterations = 100, early_stopping_rounds = 10, random_state = 42, iterations= None, loss_function = None, random_seed = None ):
        """
        Initialize the model configuration.
        
        Parameters:
        - name (str): Name of the model.
        - epochs (int): Number of training epochs.
        - batch_size (int): Batch size for training.
        - learning_rate (float): Initial learning rate for the optimizer.
        - metric (str): Evaluation metric.
        - callbacks (list): List of callbacks to use during training.
        """
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.metric = metric
        self.validation_split = validation_split
        self.callbacks = callbacks or []
        self.n_estimators = n_estimators
        self.max_leaf_nodes = max_leaf_nodes
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        self.depth = depth
        self.item_to_monitor = item_to_monitor
        self.verbose = verbose
        self.graphs = graphs
        self.num_leaves = num_leaves
        self.feature_fraction = feature_fraction
        self.num_iterations = num_iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.iterations = iterations
        self.loss_function = loss_function
        self.random_seed = random_seed

    def __repr__(self):
        """
        String representation of the configuration.
        """
        if self.model_name == 'neural_network':
            
            message = (f'''ModelConfig(model_name={self.model_name}, epochs={self.epochs}, batch_size={self.batch_size},  
                                  learning_rate={self.learning_rate}, metric={self.metric}, callbacks={len(self.callbacks)} callbacks, item_to_monitor = {self.item_to_monitor})''')
            
        elif self.model_name == 'random_forest':

            message = ModelConfig(f'model_name={self.model_name},n_estimators={self.n_estimators}, max_leaf_nodes={self.max_leaf_nodes}, n_jobs={self.n_jobs}')

        elif self.model_name == 'gbrt':

            message = ModelConfig(f'model_name={self.model_name},n_estimators={self.n_estimators}, max_depth={self.max_depth}, learning_rate={self.learning_rate}')


        return message

    def get_callbacks(self):
        """
        Return the list of callbacks.
        """
        return self.callbacks

def get_NN_model(metric_used, learning_rate):
    
    #model = keras.Sequential([layers.Dense(128, activation = 'relu',kernel_regularizer=regularizers.l2(0.001)),

    #                        layers.Dense(64, activation = 'relu', kernel_regularizer=regularizers.l2(0.001) ),
    
    #                        layers.Dense(16, activation = 'relu', kernel_regularizer=regularizers.l2(0.001) ),
                            
    #                        layers.Dense(1)])

    model = keras.Sequential([layers.Dense(64, activation = 'relu',kernel_regularizer=regularizers.l2(0.001)), 
                            
                            layers.Dense(64, activation = 'relu', kernel_regularizer=regularizers.l2(0.001) ),
                            
                            layers.Dense(1)])

    #model.compile(optimizer=keras.optimizers.RMSprop(learning_rate), loss='mse', metrics=[metric_used])
    #model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='mse', metrics=[metric_used])
    #model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss= metric_used, metrics=[tensorflow.keras.metrics.RootMeanSquaredError()])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss= 'mse', metrics=[metric_used])

    for i, layer in enumerate(model.layers):
        layer_config = layer.get_config()
        print(f"Layer {i + 1}: {layer_config['name']} with output size {layer_config['units'] if 'units' in layer_config else '1 (Output Layer)'}")

    return model

class Ensemble:

    def __init__(self, model_configs):
        
        self.model_configs = model_configs

    def fit(self, X_train ,y_train):

        for model_config in self.model_configs:

            if model_config.model_name == 'gbrt':
                
                print('Start gbrt training')

                gbrt = GradientBoostingRegressor(n_estimators=model_config.n_estimators, learning_rate=model_config.learning_rate, max_depth=model_config.max_depth, random_state=42)
                self.gbrt_fit = gbrt.fit(X_train, y_train)
                gbrt_predict = self.gbrt_fit.predict(X_train).reshape(-1, 1)
                X_train = np.hstack((X_train, gbrt_predict))
        
            elif model_config.model_name == "neural_network":
                
                print('Start neural network training')

                # Build and fit the model
                self.nn_model = get_NN_model(model_config.metric, model_config.learning_rate)

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
                
                self.nn_model.fit(X_train, y_train, epochs=model_config.epochs, batch_size=model_config.batch_size,
                                    callbacks=callbacks, verbose=1)
                self.nn_predict = self.nn_model.predict(X_train).reshape(-1, 1)
                X_train = np.hstack((X_train, self.nn_predict))  
    
            elif model_config.model_name == 'random_forest':
        
                print('Start random forest regression training')

                rfr = RandomForestRegressor(n_estimators = model_config.n_estimators, max_leaf_nodes = model_config.max_leaf_nodes, n_jobs = model_config.n_jobs)
                self.rfr_fit = rfr.fit(X_train, y_train)
                self.rfr_predict = self.rfr_fit.predict(X_train).reshape(-1, 1)
                X_train = np.hstack((X_train, self.rfr_predict))

    def predict(self, X_test):

        for model_config in self.model_configs:

            if model_config.model_name == 'gbrt':

                model_prediction = self.gbrt_fit.predict(X_test).reshape(-1, 1)
                X_test = np.hstack((X_test, model_prediction))

            elif model_config.model_name == "neural_network":

                model_prediction = self.nn_model.predict(X_test).reshape(-1, 1)
                X_test = np.hstack((X_test, model_prediction))

            elif model_config.model_name == 'random_forest':

                model_prediction = self.rfr_fit.predict(X_test).reshape(-1, 1)
                X_test = np.hstack((X_test, model_prediction))

        return model_prediction
    

class AveragingEnsemble:

    def __init__(self, model_configs):
        
        self.model_configs = model_configs

    def fit(self, X_train ,y_train, use_specfic_state = True):

        predict_train = pd.DataFrame()
        if use_specfic_state:

            random_state = 115
        
        else:

            random_state = None

        for model_config in self.model_configs:

            if model_config.model_name == 'gbrt':
                
                print('Start gbrt training')

                gbrt = GradientBoostingRegressor(n_estimators=model_config.n_estimators, learning_rate=model_config.learning_rate, max_depth=model_config.max_depth, random_state=random_state)
                self.gbrt_fit = gbrt.fit(X_train, y_train)
                self.gbrt_predict = self.gbrt_fit.predict(X_train)#.reshape(-1, 1)
                predict_train[model_config.model_name] = self.gbrt_predict
                
        
            elif model_config.model_name == "neural_network":
                
                print('Start neural network training')

                tf.random.set_seed(random_state)
                np.random.seed(random_state)
                random.seed(random_state)   

                # Build and fit the model
                self.nn_model = get_NN_model(model_config.metric, model_config.learning_rate)

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
                
                self.nn_model.fit(X_train, y_train, epochs=model_config.epochs, batch_size=model_config.batch_size,
                                    callbacks=callbacks, verbose=1)
                self.nn_predict = self.nn_model.predict(X_train)#.reshape(-1, 1)
                predict_train[model_config.model_name] = self.nn_predict
    
            elif model_config.model_name == 'random_forest':
        
                print('Start random forest regression training')

                rfr = RandomForestRegressor(n_estimators = model_config.n_estimators, max_leaf_nodes = model_config.max_leaf_nodes, n_jobs = model_config.n_jobs, random_state=random_state)
                self.rfr_fit = rfr.fit(X_train, y_train)
                self.rfr_predict = self.rfr_fit.predict(X_train)#.reshape(-1, 1)
                predict_train[model_config.model_name] = self.rfr_predict

            elif model_config.model_name == 'lightgbm':
        
                print('Start lightgbm regression training')

                train_data = lgb.Dataset(X_train, label=y_train)

                params = {
                    'metric': model_config.metric,
                    'num_leaves': model_config.num_leaves,
                    'learning_rate': model_config.learning_rate,
                    'feature_fraction': model_config.feature_fraction,
                    'num_iterations': model_config.num_iterations,
                    'max_depth': model_config.max_depth,
                    #'early_stopping_rounds': model_config.early_stopping_rounds,
                    'seed':random_state,
                    'verbose': -1
                    }

                self.lbg_model = lgb.train(train_set = train_data, params=params)

                self.lbg_predict = self.lbg_model.predict(X_train)#.reshape(-1, 1)
                predict_train[model_config.model_name] = self.lbg_predict

        return predict_train

    def predict(self, X_test):

        predict_test = pd.DataFrame()

        for model_config in self.model_configs:

            if model_config.model_name == 'gbrt':

                model_prediction = self.gbrt_fit.predict(X_test)#.reshape(-1, 1)

            elif model_config.model_name == "neural_network":

                model_prediction = self.nn_model.predict(X_test)#.reshape(-1, 1)

            elif model_config.model_name == 'random_forest':

                model_prediction = self.rfr_fit.predict(X_test)#.reshape(-1, 1)

            elif model_config.model_name == 'lightgbm':

                model_prediction = self.lbg_model.predict(X_test)#.reshape(-1, 1)

            predict_test[model_config.model_name] = model_prediction

        return predict_test

        
    

        