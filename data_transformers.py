import pandas as pd
import numpy as np
from scipy.stats import skew

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA


class ColumnDescriptor(BaseEstimator, TransformerMixin):
    def __init__(self, special_feat=None, mode = None) -> None:
        self.special_feat = special_feat
        self.output_columns_ = None  # To store the output lists
        self.mode = mode
        super().__init__()

    def fit(self, df, y=None):

        self.output_columns_ = self._process_columns(df)

        return self

    def transform(self, df):
        # Return the DataFrame unmodified (or process as needed)
        return df

    def get_output_columns(self):
        if self.output_columns_ is None:
            raise ValueError("ColumnDescriptor has not been fitted yet.")
        return self.output_columns_

    def _process_columns(self, df):
        # Example logic to generate lists (replace with your own logic)
        num_feat = [column for column in df if pd.api.types.is_numeric_dtype(df[column])]
        cat_feat = [column for column in df if not pd.api.types.is_numeric_dtype(df[column])]
    
        if self.special_feat is not None:
            for feat in self.special_feat:
    
                if feat in num_feat:  # Check if feat exists in num_feat
        
                    num_feat.remove(feat)

                    cat_feat.insert(0,feat)
        # SalePrice is not a feature but the label
        if self.mode == 'train':
            if 'SalePrice' in num_feat:
            
                num_feat.remove("SalePrice")

        # Id is not a feature
        if 'Id' in num_feat:
            
            num_feat.remove("Id")

        return num_feat, cat_feat
    

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        # If columns are dynamically set, ensure they're not None
        if self.columns is None:
            raise ValueError("Columns must be specified or dynamically set.")
        return self

    def transform(self, X):
        if self.columns is None:
            raise ValueError("Columns must be specified before transformation.")
        return X[self.columns]

class SetNAValues(BaseEstimator, TransformerMixin):
    def __init__(self, columns, ):
        super().__init__()
        """
        Initialize the class with the columns where NA values will be replaced.
        
        Parameters:
        columns (list): List of column names to modify.
        """
        self.columns = columns

    def fit(self, df, y =None):

        return self

    def transform(self, df):
        """
        Replace NA values in the specified columns with 'NotPre'.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame.
        
        Returns:
        pd.DataFrame: Modified DataFrame with 'NotPre' replacing NA values in specified columns.
        """
        df_new = df.copy()  # Create a copy of the DataFrame to avoid modifying in place
        for column in self.columns:
            if column in df_new.columns:
                df_new[column] = df_new[column].fillna('NotPre')
        return df_new
    
class DataFrameImputer(BaseEstimator, TransformerMixin):

    def __init__(self, strategy="mean"):
        self.strategy = strategy
        
    def fit(self, X, y=None):
        
        if not hasattr(self, 'statistics'):

            self.imputer = SimpleImputer(strategy=self.strategy)
            self.imputer.fit(X)
            self.statistics = pd.Series(self.imputer.statistics_, index=X.columns)  # Store statistics as a Pandas Series
        
        return self

    def transform(self, X):
        # Transform the data using SimpleImputer
        missing_cols = set(self.statistics.index) - set(X.columns)
        extra_cols = set(X.columns) - set(self.statistics.index)
        self.X = X

        missing_data = pd.DataFrame({col: self.statistics[col] for col in missing_cols}, index=X.index)

        # Concatenate the original DataFrame with the missing columns DataFrame
        X = pd.concat([X, missing_data], axis=1)
    
        # Drop extra columns that were not in the training data (optional)
        if extra_cols:
            print(f"Warning: Extra columns {extra_cols} will be ignored during transformation.")
            X = X[self.statistics.index]
        
        # Reorder columns to match the order during fitting
        X = X[self.statistics.index]

        # Transform the data using SimpleImputer
        X_imputed = self.imputer.transform(X)

        # Convert the output back into a DataFrame
        return pd.DataFrame(X_imputed, columns=self.statistics.index, index=X.index)
        
    def get_imputer_values(self):
        # Return the stored statistics
        if self.statistics is None:
            raise ValueError("Imputer has not been fitted yet.")
        return self.statistics
    
    def is_fitted(self):
        """
        Check if the imputer has been fitted by verifying if `statistics` is not None.
        """
        return self.statistics is not None
    
    def get_column_names(self):
        
        return self.X.columns
        
        
class DataFrameFeatureUnion(BaseEstimator, TransformerMixin):
    def __init__(self, transformer_list):
        self.feature_union = FeatureUnion(transformer_list)

    def fit(self, X, y=None):
        self.feature_union.fit(X, y)
        return self

    def transform(self, X):
        # Transform using FeatureUnion
        transformed = self.feature_union.transform(X)

        # If the output is not a DataFrame, convert it
        if not isinstance(transformed, pd.DataFrame):
            # Generate column names for the concatenated output
            column_names = self._get_column_names(X)
            transformed = pd.DataFrame(transformed, columns=column_names, index=X.index)
        
        return transformed

    def _get_column_names(self, X):
        column_names = []
        for name, transformer in self.feature_union.transformer_list:
            # Check if transformer is a pipeline
            if isinstance(transformer, Pipeline):
                # Iterate through steps in the pipeline
                for step_name, step_transformer in transformer.steps:
                    if hasattr(step_transformer, "get_column_names"):
                        # Try to get column names from the specific step
                        try:
                            column_names.extend(step_transformer.get_column_names())
                        except Exception as e:
                            print(f"Failed to get column names from step {step_name}: {e}")
            else:
                if hasattr(transformer, "get_column_names"):
                    # Try to get column names from the specific step
                    try:
                        column_names.extend(transformer.get_column_names())
                    except Exception as e:
                        print(f"Failed to get column names from step {step_name}: {e}")

        return column_names
        
class DataFrameScaler(BaseEstimator, TransformerMixin):
    def __init__(self, method='standardization'):
        if method not in ['standardization', 'normalization']:
            raise ValueError("Method must be either 'standardization' or 'normalization'.")
        self.method = method
        self.scaler = None
        self.statistics_ = None  # Store scaling statistics

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a Pandas DataFrame.")
        
        if self.method == 'standardization':
            self.scaler = StandardScaler()
        elif self.method == 'normalization':
            self.scaler = MinMaxScaler()
        
        self.scaler.fit(X)
        self.statistics_ = pd.DataFrame({
            'mean': self.scaler.mean_ if hasattr(self.scaler, 'mean_') else None,
            'scale': self.scaler.scale_ if hasattr(self.scaler, 'scale_') else None,
            'min': self.scaler.data_min_ if hasattr(self.scaler, 'data_min_') else None,
            'max': self.scaler.data_max_ if hasattr(self.scaler, 'data_max_') else None
        }, index=X.columns)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a Pandas DataFrame.")
        
        # Ensure all fitted columns are present in the input DataFrame
        missing_cols = self.statistics_.index.difference(X.columns)
        if not missing_cols.empty:
            # Create a DataFrame with missing columns filled with 0
            missing_data = pd.DataFrame(0, index=X.index, columns=missing_cols)

            X = pd.concat([X, missing_data], axis=1)

        # Retain only columns seen during fit and ensure the same order
        X = X[self.statistics_.index]

        # Scale the columns
        X_scaled = self.scaler.transform(X)

        # Return the scaled DataFrame
        return pd.DataFrame(X_scaled, columns=self.statistics_.index, index=X.index)
        
    def get_statistics(self):
        if self.statistics_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        return self.statistics_

    
class TransformFeatType(BaseEstimator, TransformerMixin):
    def __init__(self, columns, TypeToUse) -> None:
        super().__init__()
        self.columns = columns
        self.TypeToUse = TypeToUse

    def fit(self, df, y =None):

        return self

    def transform(self, df):

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a Pandas DataFrame.")
        
        missing_columns = [col for col in self.columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"The following columns are missing in the input DataFrame: {missing_columns}")

        df_transformed = df.copy()

        for col in self.columns:
            df_transformed[col] = df_transformed[col].astype(self.TypeToUse)

        return df_transformed

    
class OneHotEnc(BaseEstimator, TransformerMixin):
    def __init__(self,  Apply1Hot= True, columns_not_1hot = None) -> None:
        super().__init__()
        self.columns_not_1hot = columns_not_1hot
        self.Apply1Hot = Apply1Hot

    def fit(self, df, y =None):

        return self

    def transform(self, df):

        if self.columns_not_1hot is not None:

            one_hot_feat = list(filter(lambda x: x not in self.columns_not_1hot, df.columns))
        
        else:

            one_hot_feat = df.columns

        if self.Apply1Hot == True:

            df_new = df.copy()
            df_encoded = pd.get_dummies(df_new[one_hot_feat], columns=one_hot_feat, prefix=one_hot_feat).astype(int)
            
            df_new = df_new.drop(columns=one_hot_feat)
            df_combined = pd.concat([df_new, df_encoded], axis=1)
            return df_combined

        else:

            return df
    
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_age_columns = False, add_combined_num_feat = False, add_combined_cat_feat = False, add_combined_catnum_feat = False) -> None:
        super().__init__()
        self.add_age_columns = add_age_columns
        self.add_combined_num_feat = add_combined_num_feat
        self.add_combined_cat_feat = add_combined_cat_feat
        self.add_combined_catnum_feat = add_combined_catnum_feat

    def fit(self, df, y =None):

        self.df = df

        return self
    
    def transform(self, df, y = None):

        df_new = df.copy()

        # translate HouseStyle
        #df['nmbr_levels']

        if self.add_age_columns:
            current_year = 2011
            df_new['Age_Building'] = current_year - df_new['YearBuilt'].astype(int)
            df_new['Years_since_remod'] = current_year - df_new['YearRemodAdd'].astype(int)
            df_new['Years_since_garage'] = current_year - df_new['GarageYrBlt'].astype(int)
            df_new['Years_since_sold'] = current_year - df_new['YrSold'].astype(int)
        
        if self.add_combined_num_feat:

            df_new['TotalSF'] = df_new['GrLivArea'] + df_new['TotalBsmtSF']
            #df_new['TotalQual'] = df_new['OverallQual'] * df_new['TotalSF']
            df_new['TotalBath'] = df_new['FullBath'] + (0.5 * df_new['HalfBath']) + df_new['BsmtFullBath'] + (0.5 * df_new['BsmtHalfBath'])
            df_new['AvgRoomSize'] = df_new['GrLivArea'] / df_new['TotRmsAbvGrd']
            #df_new['PorchCount'] = ((df_new['OpenPorchSF'] > 0).astype(int) + (df_new['EnclosedPorch'] > 0).astype(int) + (df_new['3SsnPorch'] > 0).astype(int) + (df_new['ScreenPorch'] > 0).astype(int))
            df_new['TotalPorchSF'] = df_new['OpenPorchSF'] + df_new['EnclosedPorch'] + df_new['3SsnPorch'] + df_new['ScreenPorch']
            df_new['Garden'] = df_new['LotArea'] - (df_new['GarageArea'] + df_new['WoodDeckSF'] + df_new['TotalPorchSF']+ df_new['PoolArea'])
            #df_new['SizePerCar'] = df_new['GarageArea'] / df_new['GarageCars']
            #df_new['BsmtFinSF'] = df_new['BsmtFinSF1'] + df_new['BsmtFinSF2']
            height_mapping = {
                'Ex': 100,
                'Gd': 95,
                'TA': 85,
                'Fa': 75,
                'Po': 65,
                'NA': 1  # No basement
            }

            df_new['Height_ft'] = df_new['BsmtQual'].map(height_mapping) / 12
                   
            df_new['BasementVolume'] = df_new['Height_ft'] * df_new['TotalBsmtSF']
            df_new.loc[df_new['TotalBsmtSF'] == 0, 'BasementVolume'] = 0
            df_new.drop(columns = 'Height_ft', inplace=True)

        if self.add_combined_catnum_feat:

            dict_combination = {#'LotArea': ['LandContour', 'LotShape', 'LotConfig', 'LandSlope'],
                                #'LotFrontage': ['Street'],
                                #'MasVnrArea': ['MasVnrType'],
                                #'TotalBsmtSF': ['BsmtQual'],#['BsmtQual', 'BsmtCond', 'BsmtExposure'],
                                #'BsmtFinSF1': ['BsmtFinType1'],
                                #'BsmtFinSF2': ['BsmtFinType2'],
                                #'GrLivArea': [ 'OverallQual', 'OverallCond'],
                                #'KitchenAbvGr': ['KitchenQual'],
                                #'Fireplaces': ['FireplaceQu'],
                                #'GarageCars': ['GarageFinish', 'GarageQual', 'GarageCond'],
                                #'GarageArea': ['GarageFinish', 'GarageQual', 'GarageCond'],
                                #'PoolArea': ['PoolQC'],
                                'MiscVal': ['MiscFeature'],
                                #'Neighborhood': ['Condition1']
                                }

            for num_col, cat_cols in dict_combination.items():
                if num_col not in df_new.columns or not all(col in df_new.columns for col in cat_cols):
                    continue  # Skip if columns don't exist in the DataFrame

                for cat_col in cat_cols:
                    # Perform one-hot encoding for the categorical column
                    one_hot = pd.get_dummies(df_new[cat_col], prefix=f"{cat_col}_{num_col}")

                    # Multiply the one-hot encoding with the numerical column
                    one_hot_scaled = one_hot.mul(df_new[num_col], axis=0)

                    # Add the scaled one-hot encoded features back to the DataFrame
                    df_new = pd.concat([df_new, one_hot_scaled], axis=1)

            # Drop the key columns and original categorical columns
            columns_to_drop = list(dict_combination.keys()) + sum(dict_combination.values(), [])
            df_new.drop(columns=columns_to_drop, inplace=True)

        return df_new
    

class AttributesDeleter(BaseEstimator, TransformerMixin):
    def __init__(self, feat_to_delete = None) -> None:
        super().__init__()
        self.feat_to_delete = feat_to_delete

    def fit(self, df, y =None):
        return self
    
    def transform(self, df, y = None):

        df_new = df.copy()

        if self.feat_to_delete != None:
            df_new = df.drop(columns = self.feat_to_delete)

        return df_new
    
class DelSparseClass(BaseEstimator, TransformerMixin):
    
    def __init__(self, ToDelete, min_percentage) -> None:
        super().__init__()
        self.ToDelete = ToDelete
        self.min_percentage = min_percentage
        self.columns = None

    def fit(self, df, y =None):

        if self.ToDelete:
            
            threshold = self.min_percentage * len(df)
            self.columns = df.columns[df.sum() >= threshold]
        
        else:

            self.columns = df.columns.tolist()

        return self        

    def transform(self, df, y = None):     

        df_new = df.copy()

        for col in self.columns:
            if col not in df_new.columns:
                print(f'Column {col} is not present in the test data. Its values are set to zero')
                df_new[col] = 0  # Add the missing column with all values as 0
    
        # Filter the DataFrame to include only the desired columns
        df_filtered = df_new[self.columns]
    
        return df_filtered
        #all_columns_exist = all(col in df_new.columns for col in self.columns)

        #if not all_columns_exist:
        #    raise ValueError("Not all categorical columns present in the train df are present in the test df")
        #else:
        #    df_filtered = df_new[self.columns]
        #
        #return df_filtered
    
    #def get_column_names(self):
    #    
    #    if self.columns is None:
    #        raise ValueError("All columns are sparse.")
    #    return self.columns



class DelLowCorr(BaseEstimator, TransformerMixin):
    
    def __init__(self, ToDelete, min_corr) -> None:
        super().__init__()
        self.ToDelete = ToDelete
        self.min_corr = min_corr
        self.columns = None
        self.corr_with_target = None

    def fit(self, df, y):

        self.label = y
        df_corr = df.copy()
        df_corr.loc[:, 'SalePrice'] = self.label

        if self.ToDelete:

            correlation_matrix = df_corr.corr()
            self.corr_with_target = correlation_matrix['SalePrice'].abs().sort_values(ascending=False)
            self.columns = self.corr_with_target[self.corr_with_target > self.min_corr].index.tolist()  # Adjust threshold as needed
  
            if 'SalePrice' in self.columns:
                self.columns.remove('SalePrice')

        else:
            
            self.columns = df.columns.tolist()  # Adjust threshold as needed


        return self
    
    def transform(self, df, y = None):  
        
        df_new = df[self.columns]

        return df_new
    
    def get_correlations(self):

        return self.corr_with_target
    

class DelCollin(BaseEstimator, TransformerMixin):
    
    def __init__(self, ToDelete, min_collin = None) -> None:
        super().__init__()
        self.ToDelete = ToDelete
        self.min_collin = min_collin
        self.columns_to_delete = None

    def fit(self, df, y= None):

        if self.ToDelete:
        
            df_collin = df.copy()
            corr_matrix = df_collin.corr()  
            # Find the columns that are very positively correlated > 0.9 and very negatively correlated < -0.9
            high_corr_dict = {}
            low_corr_dict = {}

            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):  # Only check upper triangle to avoid duplicates
                    if corr_matrix.iloc[i, j] > self.min_collin:
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        high_corr_dict[(col1, col2)] = corr_matrix.iloc[i, j]
                    if corr_matrix.iloc[i, j] < -self.min_collin:
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        low_corr_dict[(col1, col2)] = corr_matrix.iloc[i, j]

            lst_feat_to_delete = []

            for tup in high_corr_dict.keys():

                lst_feat_to_delete.append(tup[1])

            for tup in low_corr_dict.keys():

                lst_feat_to_delete.append(tup[0])

            self.columns_to_delete = np.unique(lst_feat_to_delete)
            print(f'feat to delete are: {self.columns_to_delete}')

        return self
    
    def transform(self, df, y = None):

        if self.ToDelete:
        
            return df.drop(columns = self.columns_to_delete)

        else:

            return df
    
    def get_columns(self):
        
        return self.columns_to_delete


class InputNameChecker(BaseEstimator, TransformerMixin):

    def __init__(self, allowed_values) -> None:
        super().__init__()
        self.allowed_values = allowed_values

    def fit(self, df=None, y = None):

        return self
    
    def transform(self, df):
        
        invalid_values = {}

        for column, valid_values in self.allowed_values.items():
            if column in df.columns:
                valid_values_set = set(valid_values)
                # Find values in the column that are not in the allowed list
                invalid = set(df[column]) - valid_values_set
                if invalid:
                    invalid_values[column] = invalid

        # Print results
        #if invalid_values:
        #    print("Invalid values found:")
        #    for column, values in invalid_values.items():
        #        print(f"{column}: {values}")
        #else:
        #    print("All values are valid.")

        self.invalid_values = invalid_values

        transform_values = {'Exterior2nd': {'CmentBd': 'CemntBd', 'Wd Shng': 'WdShing', 'Brk Cmn': 'BrkComm'}}

        for column, mapping in transform_values.items():
            if column in df.columns:
                df[column] = df[column].replace(mapping)
        
        return df
    
    
class PCATransformer(BaseEstimator, TransformerMixin):

    def __init__(self, ToUse = False, n_components=None, columns=None):
        """
        Initialize the PCA transformer.

        Parameters:
        - n_components (int, float, or None): Number of components to keep or variance to retain.
        - columns (list or None): List of columns to apply PCA on. If None, all columns are used.
        """
        self.n_components = n_components
        self.columns = columns
        self.pca = None
        self.selected_columns_ = None
        self.ToUse = ToUse

    def fit(self, df, y=None):
        """
        Fit the PCA transformer to the data.

        Parameters:
        - X (pd.DataFrame): Input data.
        - y (ignored): Not used, present for pipeline compatibility.

        Returns:
        - self
        """
        if self.columns:
            self.selected_columns_ = self.columns
        else:
            self.selected_columns_ = df.columns

        if self.ToUse:

            self.pca = PCA(n_components=self.n_components)
            self.pca.fit(df[self.selected_columns_])

        return self

    def transform(self, df):
        """
        Apply the PCA transformation.

        Parameters:
        - df (pd.DataFrame): Input data.

        Returns:
        - pd.DataFrame: Transformed DataFrame with principal components.
        """

        if self.ToUse:

            if self.pca is None:
                raise RuntimeError("PCATransformer must be fitted before calling transform.")

            pca_data = self.pca.transform(df[self.selected_columns_])
            pca_columns = [f"PCA_{i+1}" for i in range(pca_data.shape[1])]

            # Return a DataFrame containing the PCA components
            return pd.DataFrame(pca_data, columns=pca_columns, index=df.index)
        
        else:

            return df

    def fit_transform(self, X, y=None):
        """
        Fit the PCA transformer and apply the transformation in one step.

        Parameters:
        - X (pd.DataFrame): Input data.
        - y (ignored): Not used, present for pipeline compatibility.

        Returns:
        - pd.DataFrame: Transformed DataFrame with principal components.
        """
        self.fit(X, y)
        return self.transform(X)

    def get_explained_var(self):
        """
        Returns the PCA loadings (components) as a DataFrame.

        Returns:
        - pd.DataFrame: The loadings of each original feature on the principal components.
        """
        if self.ToUse:
        
            if self.pca is None:
                raise RuntimeError("PCATransformer must be fitted before accessing loadings.")
        
            # Create a DataFrame for PCA loadings
            expl_var =  pd.Series(
                self.pca.explained_variance_ratio_,
                index=[f"PCA_{i+1}" for i in range(len(self.pca.explained_variance_ratio_))],
                name="Explained Variance Ratio")

            #expl_var .to_csv('expl_var.csv', index=False)

            return expl_var
        
        else:

            return None
    
class SkewTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, skew_threshold=0.75):
        """
        Initialize the transformer with a skewness threshold.
        
        Parameters:
        - skew_threshold: Features with skewness above this value will be log-transformed.
        """
        self.skew_threshold = skew_threshold
        self.skewed_features_ = []

    def fit(self, X, y=None):
        """
        Identify skewed features in the data.
        
        Parameters:
        - X: DataFrame, the input data.
        - y: Ignored, for compatibility with scikit-learn.
        
        Returns:
        - self
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        # Calculate skewness for numerical features
        self.skewness = X.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        self.skewed_features_ = self.skewness[self.skewness > self.skew_threshold].index.tolist()
        
        return self

    def transform(self, X):
        """
        Apply log transformation to identified skewed features.
        
        Parameters:
        - X: DataFrame, the input data.
        
        Returns:
        - X_transformed: DataFrame, the transformed data.
        """
        self.X = X.copy()
    
        # Apply log transformation
        for feature in self.skewed_features_:
            if feature in self.X.columns:
                self.X[feature] = np.log1p(self.X[feature])

        return self.X
    
    def get_skew_values(self):

        return self.skewness
    
    def get_X(self):

        return self.X


    
class DelOutlier(BaseEstimator, TransformerMixin):
    def __init__(self, ToDelete = False):
         
         self.ToDelete = ToDelete

    def fit(self, X, label=None):

        self.label = label
        
        return self

    def transform(self, X):
        # Ensure X is a DataFrame (if it's not, convert it)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Define the conditions for outliers
        conditions = (
            #((X['OverallQual'] == 10) & (self.label == 160000)) | 
            #((X['Foundation'] =='BrkTil') & (self.label == 475000)) |
            #((X['Neighborhood'] =='NoRidge') & (self.label == 755000))
            #((X['SaleCondition'] =='Abnorml') & (self.label == 745000))| # goed
            #((X['SaleCondition'] =='Abnorml') & (self.label == 34900))| # goed
            #((X['SaleCondition'] =='Abnorml') & (self.label == 35311)) beter in val niet in test
            #((X['SaleCondition'] =='Abnorml') & (self.label == 37900)) Goed
            #((X['BsmtQual'] =='Fa') & (self.label == 61000))
            ((X['TotalBsmtSF'] ==504) & (self.label == 127000))
            )
        
        # Check if ToDelete is True
        if self.ToDelete:
            # Remove rows that meet the conditions
            X = X[~conditions]
            label = self.label[~conditions]

        self.label = label  # Update the label attribute after filtering
        return X.reset_index(drop=True)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return (self.transform(X), self.label)
    
class ZeroColumnRemover(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.non_zero_columns_ = None  # To store columns that are not all zeros during fitting
        self.columns = None

    def fit(self, X, y=None):
        
        return self

    def transform(self, X, **kwargs):
        """
        Remove columns that are entirely zeros in the input data.
        
        Parameters:
        - X: DataFrame or array-like, the input data.
        
        Returns:
        - X_transformed: DataFrame with zero-only columns removed.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.non_zero_columns_ = X.columns[(X != 0).any(axis=0)]

        # Step 1: Keep only columns identified during fit
        X = X[self.non_zero_columns_]

        # Step 2: Recheck for all-zero columns in the new data
        X = X.loc[:, (X != 0).any(axis=0)]

        self.columns = X.columns.tolist()  # Adjust threshold as needed
        
        return X
    
    def get_column_names(self):
        if self.columns is None:
            raise ValueError("All columns have a low correlation with SalePrice.")
        return self.columns
    
        




