import pandas as pd
import numpy as np
from enum import Enum
from sklearn.pipeline import Pipeline


import my_functions
import metrics
import model_evaluate
import model_predict
import random_forest
from models import ModelConfig
from data_analysis import data_analysis
from scipy.spatial.distance import cdist


from data_transformers import DataFrameSelector, SetNAValues, DataFrameImputer, OneHotEnc, CombinedAttributesAdder, DataFrameFeatureUnion, DelSparseClass, DelLowCorr, TransformFeatType, DelCollin, InputNameChecker, PCATransformer, SkewTransformer, DelOutlier, DataFrameScaler, AttributesDeleter, ColumnDescriptor, ZeroColumnRemover

df_train = pd.read_csv("train.csv", na_values=['NA'], keep_default_na=False) # We do specifically na_values=['NA'] so that None isn't translated into np.nan as well
print(df_train.shape)
df_test = pd.read_csv("test.csv", na_values=['NA'], keep_default_na=False)

label = df_train['SalePrice']

use_outlier_deleter = False

if use_outlier_deleter:
    outlier_deleter = DelOutlier(ToDelete = True)
    (df_train, label) = outlier_deleter.fit_transform(df_train, label)

    log_label = np.log(label)
    log_label = log_label.reset_index(drop=True)

else:

    log_label = np.log(label)


# Set cat parameters of cat_num Pipeline
allowed_values = {'MSSubClass': [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190],
                  'MSZoning': ['A', 'C (all)', 'FV', 'I', 'RH', 'RL', 'RP', 'RM'],
                  'Street': ['Grvl', 'Pave'],
                  'Alley': ['Grvl', 'Pave', 'NotPre'],		
                  'LotShape': ['Reg', 'IR1', 'IR2', 'IR3'],       
                  'LandContour': ['Lvl', 'Bnk', 'HLS', 'Low'],		
                  'Utilities': ['AllPub', 'NoSewr', 'NoSeWa', 'ELO'],
                  'LotConfig': ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'],
                  'LandSlope': ['Gtl', 'Mod', 'Sev'],
                  'Neighborhood': ['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor', 'Edwards', 
                                    'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel', 'NAmes', 'NoRidge', 'NPkVill', 'NridgHt', 
                                    'NWAmes', 'OldTown', 'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'],
                  'Condition1': ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'],
                  'Condition2': ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'],
                  'BldgType': ['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs'],
                  'HouseStyle': ['1Story', '1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl'],
                  'OverallQual': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                  'OverallCond': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                  'RoofStyle': ['Flat', 'Gable', 'Gambrel', 'Hip', 'Mansard',  'Shed'],
                  'RoofMatl': ['ClyTile', 'CompShg', 'Membran', 'Metal', 'Roll', 'Tar&Grv', 'WdShake', 'WdShngl'],
                  'Exterior1st': ['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 
                                  'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'WdShing'],
                  'Exterior2nd': ['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 
                                  'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'WdShing'], 
                  'MasVnrType': ['BrkCmn', 'BrkFace', 'CBlock', 'None', 'Stone'],	
                  'ExterQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
                  'ExterCond': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
                  'Foundation': ['BrkTil', 'CBlock', 'PConc', 'Slab', 'Stone', 'Wood'],
                  'BsmtQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NotPre'],
                  'BsmtCond': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NotPre'],
                  'BsmtExposure': ['Gd', 'Av', 'Mn', 'No', 'NotPre'],
                  'BsmtFinType1': ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NotPre'],
                  'BsmtFinType2': ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NotPre'],
                  'Heating': ['Floor', 'GasA', 'GasW', 'Grav', 'OthW', 'Wall'],
                  'HeatingQC': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
                  'CentralAir': ['N', 'Y'],
		          'Electrical': ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'],		
                  'KitchenQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
                  'Functional': ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'],
                  'FireplaceQu': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NotPre'],
                  'GarageType': ['2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd', 'NotPre'],
                  'GarageFinish': ['Fin', 'RFn', 'Unf', 'NotPre'],
                  'GarageQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NotPre'],
                  'GarageCond': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NotPre'],
                  'PavedDrive': ['Y', 'P', 'N'],
                  'PoolQC': ['Ex', 'Gd', 'TA', 'Fa', 'NotPre'],		
                  'Fence': ['GdPrv', 'MnPrv', 'GdWo', 'MnWw', 'NotPre'],
                  'MiscFeature': ['Elev', 'Gar2', 'Othr', 'Shed', 'TenC', 'NotPre'],
                  'SaleType': ['WD', 'CWD', 'VWD', 'New', 'COD', 'Con', 'ConLw', 'ConLI', 'ConLD', 'Oth'],
                  'SaleCondition': ['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial'], 
                    }

correct_nan_feat = ['Alley', 'BsmtQual', 'BsmtCond','BsmtExposure','BsmtFinType1', 'BsmtFinType2','FireplaceQu','GarageQual', 'GarageCond', 'GarageType', 'GarageFinish','MiscFeature', 'PoolQC', 'Fence']

#Prework so some features are combined. This is FEATURE GENERATION
def create_preprocessing_pipeline(feat_to_delete, correct_nan_feat, allowed_values, special_feat, mode='train'):
    return Pipeline([
        ('attr_deleter', AttributesDeleter(feat_to_delete=feat_to_delete)),
        ('na_setter', SetNAValues(correct_nan_feat)),
        ('input_checker', InputNameChecker(allowed_values)),
        ('descriptor', ColumnDescriptor(special_feat, mode=mode))
    ])

def create_feature_engineering_pipeline(add_combined_num_feat, add_combined_catnum_feat, add_age_columns, special_feat, mode='train'):
    return Pipeline([
        ('attr_adder', CombinedAttributesAdder(
            add_combined_num_feat=add_combined_num_feat,
            add_combined_catnum_feat=add_combined_catnum_feat,
            add_age_columns=add_age_columns
        )),
        ('descriptor', ColumnDescriptor(special_feat, mode=mode))
    ])

feat_to_delete = ['MSSubClass']
special_feat = ['OverallQual', 'OverallCond']

preprocessing_pipeline = create_preprocessing_pipeline(
    feat_to_delete=feat_to_delete, 
    correct_nan_feat=correct_nan_feat, 
    allowed_values=allowed_values, 
    special_feat=special_feat, 
    mode='train'
)

df_train = preprocessing_pipeline.fit_transform(df_train)
initial_num_feat_train, initial_cat_feat_train = preprocessing_pipeline.named_steps['descriptor'].get_output_columns()

cat_imputer_pipeline = Pipeline([('selector', DataFrameSelector(initial_cat_feat_train)),
                                ('imputer', DataFrameImputer(strategy='most_frequent')),
                                ])

df_train_cat = cat_imputer_pipeline.fit_transform(df_train)
fitted_cat_imputer = cat_imputer_pipeline.named_steps['imputer']

num_imputer_pipeline = Pipeline([('selector', DataFrameSelector(initial_num_feat_train)),
                                ('imputer', DataFrameImputer(strategy='mean')),
                                ])

df_train_num = num_imputer_pipeline.fit_transform(df_train)
fitted_num_imputer = num_imputer_pipeline.named_steps['imputer']

df_train = pd.concat([df_train_num, df_train_cat], axis=1)

add_combined_num_feat=True
add_combined_catnum_feat=True
add_age_columns=True

feature_engineering_pipeline = create_feature_engineering_pipeline(
    add_age_columns =add_age_columns,
    add_combined_num_feat = add_combined_num_feat,
    add_combined_catnum_feat = add_combined_catnum_feat,
    special_feat=special_feat,
    mode='train'
)
df_train = feature_engineering_pipeline.fit_transform(df_train)
num_feat_train, cat_feat_train = feature_engineering_pipeline.named_steps['descriptor'].get_output_columns()

# Set cat parameters of cat train Pipeline
Apply1Hot = False
SparseToDelete = False
CatCorrToDelete = False

cat_train_pipeline = Pipeline([
                            ('selector', DataFrameSelector(cat_feat_train)),
                            ('one_hot', OneHotEnc(Apply1Hot = Apply1Hot)),
                            ('sparse', DelSparseClass(ToDelete = SparseToDelete, min_percentage = 0.04)), # This has to be True since else some columns in the train set are not present in the test set. These have to be deleted anyways
                            ('corr_cat_check', DelLowCorr(ToDelete = CatCorrToDelete, min_corr = 0.05)),
                            ('zeros_remover', ZeroColumnRemover()),
                            ])


# Set numerical parameters of Pipeline
NumCorrToDelete = False
methods = ['standardization', 'normalization']
ScalerMethodUsed = methods[1]

num_train_pipeline = Pipeline([
                        ('selector', DataFrameSelector(num_feat_train)),
                        ('skew_transform', SkewTransformer(skew_threshold = 1.0)),
                        ('scaler', DataFrameScaler(method = ScalerMethodUsed)),
                        ('corr_num_check', DelLowCorr(ToDelete = NumCorrToDelete, min_corr = 0.05)),
                        ('zeros_remover', ZeroColumnRemover()),
                        ])

# Set parameters of the combined Pipeline
CollinToDelete = False
PcaToUse = False

combined_train_pipeline = Pipeline([
    ('feature_union', DataFrameFeatureUnion([
        ('num_pipeline', num_train_pipeline),
        ('cat_pipeline', cat_train_pipeline),
    ])),
    ('collin_check', DelCollin(ToDelete = CollinToDelete, min_collin = 0.998)),
    ('pca', PCATransformer(ToUse = PcaToUse, n_components= 0.99)),  # Optional final transformation step
])

df_train_prep = combined_train_pipeline.fit_transform(df_train, label)
df_train_prep = df_train_prep.reset_index(drop=True)
df_train_prep = my_functions.set_column_type(df_train_prep)

df_train_final = pd.concat([df_train_prep, log_label], axis=1)

print('The final train data output is')
print(df_train_final)
print()

fitted_cat_sparser = combined_train_pipeline.named_steps['feature_union'].feature_union.transformer_list[1][1].named_steps['sparse']
fitted_cat_corr_check = combined_train_pipeline.named_steps['feature_union'].feature_union.transformer_list[1][1].named_steps['corr_cat_check']

fitted_num_skew = combined_train_pipeline.named_steps['feature_union'].feature_union.transformer_list[0][1].named_steps['skew_transform']
fitted_num_scaler = combined_train_pipeline.named_steps['feature_union'].feature_union.transformer_list[0][1].named_steps['scaler']
fitted_num_corr_check = combined_train_pipeline.named_steps['feature_union'].feature_union.transformer_list[0][1].named_steps['corr_num_check']

print_correlations = False

if print_correlations:
    cat_correlations = fitted_cat_corr_check.get_correlations()
    pd.set_option('display.max_rows', None)
    print(f'The corr of num features with SalePrice are {cat_correlations}')
    print()
    
    num_correlations = fitted_num_corr_check.get_correlations()
    print(f'The corr of num features with SalePrice are {num_correlations}')
    pd.reset_option('display.max_rows')

fitted_collin = combined_train_pipeline.named_steps['collin_check']
collin_columns = fitted_collin.get_columns()
fitted_PCA = combined_train_pipeline.named_steps['pca']
explained_var = fitted_PCA.get_explained_var()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
#TEST THE PIPELINE
#partial_pipeline = Pipeline(feature_engineering_pipeline.steps[:1])
#df_partial = partial_pipeline.fit_transform(df_train, label)
#print(df_partial)
#df_partial.to_csv('df_partial.csv', index=False)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print('Start of the test data prep')
print()
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
preprocessing_pipeline = create_preprocessing_pipeline(
    feat_to_delete=feat_to_delete, 
    correct_nan_feat=correct_nan_feat, 
    allowed_values=allowed_values, 
    special_feat=special_feat, 
    mode='test'
)

df_test = preprocessing_pipeline.fit_transform(df_test)

initial_num_feat_test, initial_cat_feat_test = preprocessing_pipeline.named_steps['descriptor'].get_output_columns()

cat_imputer_pipeline = Pipeline([('selector', DataFrameSelector(initial_cat_feat_test)),
                                ('imputer', fitted_cat_imputer),
                                ])

num_imputer_pipeline = Pipeline([('selector', DataFrameSelector(initial_num_feat_test)),
                                ('imputer', fitted_num_imputer),
                                ])

df_cat = cat_imputer_pipeline.fit_transform(df_test)
df_num = num_imputer_pipeline.fit_transform(df_test)
df_test = pd.concat([df_cat, df_num], axis = 1)

test_feature_engineering_pipeline = create_feature_engineering_pipeline(
    add_age_columns =add_age_columns,
    add_combined_num_feat = add_combined_num_feat,
    add_combined_catnum_feat = add_combined_catnum_feat,
    special_feat=special_feat,
    mode='test')

df_test = test_feature_engineering_pipeline.fit_transform(df_test)
num_feat_test, cat_feat_test = test_feature_engineering_pipeline.named_steps['descriptor'].get_output_columns()

cat_test_pipeline = Pipeline([
                            ('selector', DataFrameSelector(cat_feat_test)),
                            ('one_hot', OneHotEnc(Apply1Hot = Apply1Hot)),
                            ('sparse', fitted_cat_sparser), # here all none existent columns are added and given the value 0
                            ('corr_cat_check', fitted_cat_corr_check),
                            ('zeros_remover', ZeroColumnRemover()),
                            ])

num_test_pipeline = Pipeline([
                        ('selector', DataFrameSelector(num_feat_test)),
                        ('skew_transform', fitted_num_skew), # because of the possible combination between num and cat features, its possible that some columns in test are not in train. These have to be taken out
                        ('scaler', fitted_num_scaler), # because of the possible combination between num and cat features, its possible that some columns in test are not in train. These have to be taken out
                        ('corr_num_check', fitted_num_corr_check),
                        ('zeros_remover', ZeroColumnRemover()),
                        ])

combined_test_pipeline = Pipeline([
    ('feature_union', DataFrameFeatureUnion([
        ('num_pipeline', num_test_pipeline),
        ('cat_pipeline', cat_test_pipeline),
    ])),
    ('collin_check', fitted_collin),  # Optional final transformation step
    ('pca', fitted_PCA)
])

df_test_prep = combined_test_pipeline.transform(df_test)
# The train and test df need to have the same amount of columns
df_test_prep = my_functions.align_columns(df_train_prep, df_test_prep)
df_test_prep = my_functions.set_column_type(df_test_prep)

print('The test df after transformation is')
print(df_test_prep)
print()

df_train_prep.to_csv('df_train_prep.csv', index=False)
df_test_prep.to_csv('df_test_prep.csv', index=False)


#========================================================================================
#COMPARE TEST AND TRAIN DATASET TO FIND OPTIMAL ROWS
# Implementation Using Euclidean Distance

do_distance_analysis = False

if do_distance_analysis:
    df_1460 = df_train_prep.copy()
    df_1459 = df_test_prep.copy()

    for col, weight in cat_correlations.items():
        if col in df_1460.columns:
            df_1460[col] *= weight
            df_1459[col] *= weight

    for col, weight in num_correlations.items():
        if col in df_1460.columns:
            df_1460[col] *= weight
            df_1459[col] *= weight

    distances = cdist(df_1460.values, df_1459.values, metric='euclidean')
    min_distances = distances.min(axis=1)

    worst_match_index = np.argmax(min_distances)
    worst_match_distance = min_distances[worst_match_index]
    closest_row_index = np.argmin(distances[worst_match_index])

    print(f"The row {worst_match_index} in the 1460-row dataset is the worst match.")
    print(f"Distance to closest row in 1459-row dataset: {worst_match_distance}")
    print(f"The row {closest_row_index} in the 1459-row dataset is the closest match.")

    worst_row = df_1460.iloc[worst_match_index]
    print('the worst row is')
    print(worst_row)
    print()
    closest_row_index = np.argmin(distances[worst_match_index])
    closest_row = df_1459.iloc[closest_row_index]
    print('the closest row is')
    print(closest_row)
    print()


    # Compute feature-wise contributions to the Euclidean distance
    feature_differences = (worst_row - closest_row).abs()
    feature_contributions = feature_differences * feature_differences  # Squared contributions
    weighted_contributions = feature_contributions * np.array([
        cat_correlations.get(col, num_correlations.get(col, 1)) for col in df_1460.columns
    ])

    # Rank features by their contribution
    contribution_df = pd.DataFrame({
        'Feature': df_1460.columns,
        'Difference': feature_differences,
        'Weighted Contribution': weighted_contributions
    }).sort_values(by='Weighted Contribution', ascending=False)

    print(contribution_df)
#============================================================================================
# OTher model to use Lasso, ElasticNet, KernelRidge, Catboost, LGB and XGB in the ensembl
# Other way of ensembles, weighted average, voting ensembles
# Hyper parameter tuning
#---------------------------------------------------------------------------------------
# TRAIN THE MODELS
#---------------------------------------------------------------------------------------
class EVALUATIONMode(Enum):
    NONE = 0
    EVAL = 1
    EVAL_K_FOLD = 2
    ENSEMBLE = 3
    ENSEMBLE_K_FOLD = 4
    SHAP = 5

class PREDICTIONMode(Enum):
    NONE = 0
    PREDICT = 1
    ENSEMBLE = 2
    GBRT_K_FOLD = 3
    SHAP = 4
    AVG = 5


# Set the desired training mode here
evaluation_mode = EVALUATIONMode.EVAL_K_FOLD
prediction_mode = PREDICTIONMode.NONE

# evaluate Config
item_to_monitor = 'loss'
batch_size = 16
random_state = 115

nn_model_config = ModelConfig(
        model_name = 'neural_network',
        metric = metrics.root_mean_squared_error,
        epochs = 600,
        batch_size = batch_size,
        learning_rate = 0.005,
        item_to_monitor = item_to_monitor,
        callbacks = {'StopAtMinLR': {'monitor': item_to_monitor, 'min_lr':1e-6, 'patience': 1, 'verbose': 1},
                     'ReduceLROnPlateau': {'monitor': item_to_monitor, 'factor': 0.5, 'patience': 10, 'min_lr': 1e-7, 'verbose': 1},
                     'AdjustBatchSizeOnLR': {'initial_batch_size': batch_size, 'lr_threshold': 0.0005, 'new_batch_size': 64}},
        random_state = random_state,
        verbose = 1,)

rfr_model_config = ModelConfig(
        model_name = 'random_forest',
        n_estimators = 500, 
        max_leaf_nodes = 32,
        n_jobs = -1,
        random_state = random_state,
        )

gbrt_model_config = ModelConfig(
        model_name = 'gbrt',
        max_depth = 4, 
        n_estimators = 500,
        learning_rate = 0.05,
        random_state = random_state,
        )

lightgbm_model_config = ModelConfig(
        model_name = 'lightgbm',
        metric = 'rmse',
        num_leaves = 31,
        learning_rate = 0.05,
        feature_fraction = 0.9,
        num_iterations = 1000,
        max_depth = 4,
        early_stopping_rounds = 10,
        random_state = random_state,
        )

catboost_model_config = ModelConfig(
        model_name='catboost',
        iterations = 1000,
        depth = 5,
        learning_rate = 0.1, 
        loss_function='RMSE', 
        random_seed = random_state,       
)

#-------------------------------------------------------------------------------------------------------------------------
# POSSBLE MODELS
model_configs = [nn_model_config, rfr_model_config, gbrt_model_config, lightgbm_model_config, catboost_model_config]
# WHICH SINGLE MODEL DO YOU WANT TO USE FOR EVALUATION OR PREDICTION
model_to_use = model_configs[4]
# WHICH MODELS DO YOU WANT TO USE in ENSEMBLE FOR EVALUATION OR PREDICTION / CANNOT USE MODELS MULTIPLE TIMES
ensemble_models = [model_configs[2], model_configs[1], model_configs[3], model_configs[0]]

create_file = True

def evaluate(df_train, K, model_config):

    rmse_train, rmse_val = model_evaluate.evaluate_model(df_train, K, model_config)
    print(f'rmse_train is {rmse_train}')
    print(f'rmse_val is {rmse_val}')

def evaluate_ensemble(df_train, K, model_configs):

    rmse_train, rmse_val = model_evaluate.evaluate_ensemble(df_train, K, model_configs)
    print(f'rmse_train is {rmse_train}')
    print(f'rmse_val is {rmse_val}')

def evaluate_shap(df_train, model_configs):

    predictions = model_evaluate.evaluate_shap(df_train, model_configs)

    if create_file:

        my_functions.submission_file(predictions) 


def predict(df_train, df_test, model_config):

    predictions = model_predict.predict_model(df_train, df_test, model_config)
    print(f'The predictions are: {predictions[:50]}')

    if create_file:

        my_functions.submission_file(predictions)

def predict_ensemble(df_train, df_test, model_configs):

    predictions = model_predict.predict_ensemble(df_train, df_test, model_configs)
    print(f'The predictions are: {predictions[:50]}')

    if create_file:

        my_functions.submission_file(predictions)

def predict_gbrt_k_fold(df_train, df_test, model_config, K):

    predictions = model_predict.predict_GBRT_k_fold(df_train, df_test, model_config, K)
    print(f'The predictions are: {predictions[:50]}')

    if create_file:

        my_functions.submission_file(predictions)

def predict_shap(df_train, df_test, model_config):

    shap_values = model_predict.predict_shap(df_train, df_test, model_config)
    mean_abs_shap_values = pd.DataFrame({
    'Feature': df_test.columns,  # Replace with the dataset's feature names
    'Mean Absolute SHAP Value': np.abs(shap_values.values).mean(axis=0)
    }).sort_values(by='Mean Absolute SHAP Value', ascending=False)

    # Get the feature names in the order they appear in the summary plot
    ordered_feature_names = mean_abs_shap_values['Feature'].tolist()

    print("Feature names in the order they appear in the SHAP summary plot:")
    print(ordered_feature_names)

def predict_avg(df_train, df_test, model_configs, use_specfic_state):

    predictions = model_predict.predict_avg(df_train, df_test, model_configs, use_specfic_state)
    print(f'The predictions are: {predictions[:50]}')

    if create_file:

        my_functions.submission_file(predictions)
#-----------------------------------------------------------------------------------------------------------------------

if evaluation_mode != EVALUATIONMode.NONE:
    if evaluation_mode == EVALUATIONMode.EVAL:
        evaluate(df_train_final, 1, model_to_use)
    if evaluation_mode == EVALUATIONMode.EVAL_K_FOLD:
        evaluate(df_train_final, 4, model_to_use)
    elif evaluation_mode == EVALUATIONMode.ENSEMBLE:
        evaluate_ensemble(df_train_final, 1, ensemble_models)
    elif evaluation_mode == EVALUATIONMode.ENSEMBLE_K_FOLD:
        evaluate_ensemble(df_train_final, 4, ensemble_models)
    elif evaluation_mode == EVALUATIONMode.SHAP:
        evaluate_shap(df_train_final, model_to_use)
elif prediction_mode != PREDICTIONMode.NONE:
    if prediction_mode == PREDICTIONMode.PREDICT:
        predict(df_train_final, df_test_prep, model_to_use)
    elif prediction_mode == PREDICTIONMode.ENSEMBLE:
        predict_ensemble(df_train_final, df_test_prep, ensemble_models)
    elif prediction_mode == PREDICTIONMode.GBRT_K_FOLD:
        predict_gbrt_k_fold(df_train_final, df_test_prep, gbrt_model_config, 4)
    elif prediction_mode == PREDICTIONMode.SHAP:
        predict_shap(df_train_final, df_test_prep, model_to_use)
    elif prediction_mode == PREDICTIONMode.AVG:
        predict_avg(df_train_final, df_test_prep, ensemble_models, use_specfic_state = True)
    else:
        print("Invalid training mode selected.")
else:
    print("No training selected.")

