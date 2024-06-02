import subprocess
import sys
subprocess.call([sys.executable, '-m', 'pip', 'install', 'category_encoders'])  #replaces !pip install
import category_encoders as ce
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import f1_score  
import pandas as pd
import numpy as np
import sklearn
from sklearn.impute import KNNImputer
sklearn.set_config(transform_output="pandas")  #says pass pandas tables through pipeline instead of numpy matrices
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from xgboost import XGBClassifier  #using sklearn compatible version
xgb_model = XGBClassifier(random_state=1234, objective='binary:logistic', eval_metric='auc')


#Custom Mapping Transformer
class CustomMappingTransformer(BaseEstimator, TransformerMixin):

  def __init__(self, mapping_column, mapping_dict:dict):
    assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return self

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?

    #Set up for producing warnings. First have to rework nan values to allow set operations to work.
    #In particular, the conversion of a column to a Series, e.g., X[self.mapping_column], transforms nan values in strange ways that screw up set differencing.
    #Strategy is to convert empty values to a string then the string back to np.nan
    placeholder = "NaN"
    column_values = X[self.mapping_column].fillna(placeholder).tolist()  #convert all nan values to the string "NaN" in new list
    column_values = [np.nan if v == placeholder else v for v in column_values]  #now convert back to np.nan
    keys_values = self.mapping_dict.keys()

    column_set = set(column_values)  #without the conversion above, the set will fail to have np.nan values where they should be.
    keys_set = set(keys_values)      #this will have np.nan values where they should be so no conversion necessary.

    #check to see if all keys are contained in column.
    keys_not_found = keys_set - column_set
    if keys_not_found:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] these mapping keys do not appear in the column: {keys_not_found}\n")

    #check to see if some keys are absent
    keys_absent = column_set -  keys_set
    if keys_absent:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] these values in the column do not contain corresponding mapping keys: {keys_absent}\n")

    #do actual mapping
    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    #self.fit(X,y)
    result = self.transform(X)
    return result

#One Hot Encoding Transformer
class CustomOHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=False):
    self.target_column = target_column
    self.dummy_na = dummy_na
    self.drop_first = drop_first

  #fill in the rest below
  def fit(self, X, y=None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return self

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected DataFrame but got {type(X)} instead.'
    # Check if the target column exists in the DataFrame
    if self.target_column not in X.columns:
      error_message = f"\nError: {self.__class__.__name__} - The target column '{self.target_column}' does not exist in the DataFrame.\n"
      raise AssertionError(error_message)

    X_ = X.copy()
    # Perform one-hot encoding on the target column
    X_ = pd.get_dummies(X, columns=[self.target_column], dummy_na=self.dummy_na, drop_first=self.drop_first)

    return X_

  def fit_transform(self, X, y=None):
    # self.fit(X, y)
    result = self.transform(X)
    return result
class CustomSigma3Transformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column):
    self.target_column = target_column
    self.boundaries = None

  def fit(self, X, y=None):
    assert isinstance(X, pd.DataFrame), "Input data must be a pandas DataFrame"
    assert self.target_column in X.columns, f'Misspelling "{self.target_column}".'

    mean = X[self.target_column].mean()
    std = X[self.target_column].std()

    lower_boundary = mean - 3 * std
    upper_boundary = mean + 3 * std

    self.boundaries = (lower_boundary, upper_boundary)
    return self

  def transform(self, X):
    assert self.boundaries is not None, f'"{self.__class__.__name__}": Missing fit.'
    lower_boundary, upper_boundary = self.boundaries
    X_ = X.copy()
    X_[self.target_column] = np.clip(X_[self.target_column], lower_boundary, upper_boundary)
    return X_.reset_index(drop=True)

  def fit_transform(self, X, y=None):
    self.fit(X)
    return self.transform(X)

#Tukey Transformer
class CustomTukeyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, fence='outer'):
        assert fence in ['inner', 'outer'], "Fence must be 'inner' or 'outer'"
        self.target_column = target_column
        self.fence = fence
        self.boundaries = None

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame), "Input data must be a pandas DataFrame"
        assert self.target_column in X.columns, f'Misspelling "{self.target_column}".'

        q1 = X[self.target_column].quantile(0.25)
        q3 = X[self.target_column].quantile(0.75)
        iqr = q3 - q1

        if self.fence == 'outer':
          outer_low = q1 - 3.0 * iqr
          outer_high = q3 + 3.0 * iqr
          self.boundaries = (outer_low, outer_high)
        else:
          inner_low = q1 - 1.5 * iqr
          inner_high = q3 + 1.5 * iqr
          self.boundaries = (inner_low, inner_high)


        return self

    def transform(self, X):
        assert self.boundaries is not None, f'"{self.__class__.__name__}": Missing fit.'

        lower_boundary, upper_boundary = self.boundaries
        X_ = X.copy()
        X_[self.target_column] = X_[self.target_column].clip(lower=lower_boundary, upper=upper_boundary)
        return X_
        #return X_.reset_index()
        

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

#Robust Scaler Transformer
class CustomRobustTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column):
    self.column = column
    self.median_ = None
    self.iqr_ = None

  def fit(self, X, y=None):
    # Calculate the median and IQR for the specified column
    self.median_ = X[self.column].median()
    Q1 = X[self.column].quantile(0.25)
    Q3 = X[self.column].quantile(0.75)
    self.iqr_ = Q3 - Q1
    return self

  def transform(self, X):
    # Apply the Robust Transformer transformation to the specified column
    X_ = X.copy()
    X_[self.column] = (X_[self.column] - self.median_) / self.iqr_
    #X_[self.column].fillna(0, inplace=True)  # Fill NaN values with 0
    return X_
    #return X_.reset_index(drop=True)

  def fit_transform(self, X, y=None):
    # Fit and transform the specified column
    self.fit(X)
    return self.transform(X)

#Function to find the random speed to test and train data set
def find_random_state(features_df, labels, n=200):
  model = KNeighborsClassifier(n_neighbors=5)  #k = 5
  var = []  
  for i in range(1, n):
    train_X, test_X, train_y, test_y = train_test_split(features_df, labels, test_size=0.2, random_state=i, shuffle=True, stratify=labels)
    model.fit(train_X, train_y)  
    train_pred = model.predict(train_X)           
    test_pred = model.predict(test_X)             
    train_f1 = f1_score(train_y, train_pred)   
    test_f1 = f1_score(test_y, test_pred)      
    f1_ratio = test_f1/train_f1          
    var.append(f1_ratio)

  rs_value = sum(var)/len(var)  #average ratio 
  idx = np.array(abs(var - rs_value)).argmin()  #index of the smallest value
  return idx



#def numpy_converter(X, y=None):
  #  assert isinstance(X, pd.core.frame.DataFrame)
   # return X.to_numpy()
#Logistix Regression CV
def dataset_setup(original_table, label_column_name, the_transformer, rs, ts=.2):
    # Extract features and labels
    features = original_table.drop(columns=[label_column_name])
    labels = original_table[label_column_name]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=ts, shuffle=True, random_state=rs, stratify=labels)

    # Apply the transformer to the training and testing data
    X_train_transformed = the_transformer.fit_transform(X_train, y_train)
    X_test_transformed = the_transformer.transform(X_test)

    # Convert the data to numpy arrays
    X_train_numpy = X_train_transformed.to_numpy()
    X_test_numpy = X_test_transformed.to_numpy()
    y_train_numpy = np.array(y_train)
    y_test_numpy = np.array(y_test)

    return X_train_numpy, X_test_numpy, y_train_numpy, y_test_numpy



# Function to calculate precison, recall, f1score and accuracy for different thresholds
def threshold_results(thresh_list, actuals, predicted):
  result_df = pd.DataFrame(columns=['threshold', 'precision', 'recall', 'f1', 'accuracy'])
  for t in thresh_list:
    yhat = [1 if v >=t else 0 for v in predicted]
    #note: where TP=0, the Precision and Recall both become 0. And I am saying return 0 in that case.
    precision = precision_score(actuals, yhat, zero_division=0)
    recall = recall_score(actuals, yhat, zero_division=0)
    f1 = f1_score(actuals, yhat)
    accuracy = accuracy_score(actuals, yhat)
    result_df.loc[len(result_df)] = {'threshold':t, 'precision':precision, 'recall':recall, 'f1':f1, 'accuracy':accuracy}

  result_df = result_df.round(2)

  #Next bit fancies up table for printing. See https://betterdatascience.com/style-pandas-dataframes/
  #Note that fancy_df is not really a dataframe. More like a printable object.
  headers = {
    "selector": "th:not(.index_name)",
    "props": "background-color: #800000; color: white; text-align: center"
  }
  properties = {"border": "1px solid black", "width": "65px", "text-align": "center"}

  fancy_df = result_df.style.format(precision=2).set_properties(**properties).set_table_styles([headers])
  return (result_df, fancy_df)

#Halving Search
def halving_search(model, grid, x_train, y_train, factor=2, min_resources="exhaust", scoring='roc_auc'):

    halving_cv = HalvingGridSearchCV(
    model, grid,  #our model and the parameter combos we want to try
    scoring= scoring,  #from chapter 10
    n_jobs=-1,  #use all available cpus
    min_resources = min_resources,  #"exhaust" sets this to 20, which is non-optimal. 
    factor=factor,  #double samples and take top half of combos on each iteration
    cv=5, random_state=1234,
    refit=True,  #remembers the best combo and gives us back that model already trained and ready for testing
)

    grid_result = halving_cv.fit(x_train, y_train)
    return grid_result
