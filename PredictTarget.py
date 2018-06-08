
# coding: utf-8

# In[2]:


import sklearn
sklearn.__version__


# In[3]:


# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals


# In[4]:


import pyodbc
import pandas as pd
import sqlalchemy
import urllib
import numpy as np


from sklearn.preprocessing import Imputer
#from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import PowerTransformer
#from sklearn.preprocessing import power_transform
from scipy.stats import boxcox
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error


# In[5]:

connectionstring = "myserver"
sql = "select * from mytable"

cnxn = pyodbc.connect(connectionstring)
df_tpo = pd.read_sql(sql, cnxn)


# In[6]:


# Definition of the CategoricalEncoder class, copied from PR #9151.
# Just run this cell, or copy it to your code, do not try to understand it (yet).

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as a numeric array.
    The input to this transformer should be a matrix of integers or strings,
    denoting the values taken on by categorical (discrete) features.
    The features can be encoded using a one-hot aka one-of-K scheme
    (``encoding='onehot'``, the default) or converted to ordinal integers
    (``encoding='ordinal'``).
    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.
    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
    Parameters
    ----------
    encoding : str, 'onehot', 'onehot-dense' or 'ordinal'
        The type of encoding to use (default is 'onehot'):
        - 'onehot': encode the features using a one-hot aka one-of-K scheme
          (or also called 'dummy' encoding). This creates a binary column for
          each category and returns a sparse matrix.
        - 'onehot-dense': the same as 'onehot' but returns a dense array
          instead of a sparse matrix.
        - 'ordinal': encode the features as ordinal integers. This results in
          a single column of integers (0 to n_categories - 1) per feature.
    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:
        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories are sorted before encoding the data
          (used categories can be found in the ``categories_`` attribute).
    dtype : number type, default np.float64
        Desired dtype of output.
    handle_unknown : 'error' (default) or 'ignore'
        Whether to raise an error or ignore if a unknown categorical feature is
        present during transform (default is to raise). When this is parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros.
        Ignoring unknown categories is not supported for
        ``encoding='ordinal'``.
    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting. When
        categories were specified manually, this holds the sorted categories
        (in order corresponding with output of `transform`).
    Examples
    --------
    Given a dataset with three features and two samples, we let the encoder
    find the maximum value per feature and transform the data to a binary
    one-hot encoding.
    >>> from sklearn.preprocessing import CategoricalEncoder
    >>> enc = CategoricalEncoder(handle_unknown='ignore')
    >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
    ... # doctest: +ELLIPSIS
    CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,
              encoding='onehot', handle_unknown='ignore')
    >>> enc.transform([[0, 1, 1], [1, 0, 4]]).toarray()
    array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]])
    See also
    --------
    sklearn.preprocessing.OneHotEncoder : performs a one-hot encoding of
      integer ordinal features. The ``OneHotEncoder assumes`` that input
      features take on values in the range ``[0, max(feature)]`` instead of
      using the unique values.
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    """

    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_feature]
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """

        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        """Transform X using one-hot encoding.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.
        """
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out


# In[7]:


from sklearn.base import BaseEstimator, TransformerMixin

# Create a class to select numerical or categorical columns
# since Scikit-Learn doesn't handle DataFrames yet


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


# In[8]:


from sklearn.base import BaseEstimator, TransformerMixin


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):  # no *args or **kargs
        """
        Called when initializing the classifier
        """

    def fit(self, X, y=None):
        self.factor = 1 - X.min(axis=0)
        return self  # nothing else to do

    def transform(self, X, y=None):
        X += self.factor

        return np.log(X)


# In[9]:


numerical_cols = ['col1', 'col2', 'col3']
categorical_cols = ['col4', 'col5', 'col6']

num_attribs = numerical_cols
cat_attribs = categorical_cols


# In[10]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer


num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy="median")),
    ('tolog', LogTransformer()),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', CategoricalEncoder(encoding="onehot-dense", handle_unknown='ignore')),
])


# In[11]:


from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])


# In[12]:


df_subset = df_tpo.dropna(subset=['Target']).copy()

df_subset[df_subset['Target'] > 200] = 200
df_subset[df_subset['Target'] < -100] = -100

df_cat = df_subset[cat_attribs].astype(str).copy()
df_cat = df_cat.fillna('Missing')

df_num = df_subset[num_attribs]

df_label = df_subset[['Target']]

df_trim = pd.concat([df_cat, df_num, df_label], axis=1)


# In[13]:


train_set, test_set = train_test_split(df_trim, test_size=0.2)


# In[14]:


full_pipeline.fit_transform(df_trim)

train_labels = train_set["Target"].copy()
train_features = full_pipeline.transform(train_set)

test_labels = test_set["Target"].copy()
test_features = full_pipeline.transform(test_set)


# In[27]:


train_features.shape


# In[15]:


cat_encoder = cat_pipeline.named_steps["cat_encoder"]
cat_one_hot_attribs = list(cat_encoder.categories_)
attributes = num_attribs + cat_one_hot_attribs


# In[16]:


cat_one_hot_attribs


# In[17]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# In[18]:


from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split

lin_reg = LinearRegression()
lin_reg.fit(train_features, train_labels)
print("R2 train", lin_reg.score(train_features, train_labels))

train_predictions = lin_reg.predict(train_features)
lin_mse = mean_squared_error(train_labels, train_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

# cross validation
lin_scores = cross_val_score(lin_reg, train_features, train_labels,
                             scoring="mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
print("R2 crossval", lin_reg.score(train_features, train_labels))
print("R2 test", lin_reg.score(test_features, test_labels))


# In[19]:


# Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)
gbrt.fit(train_features, train_labels)
print("R2 = " + str(gbrt.score(train_features, train_labels)))

train_predictions = gbrt.predict(train_features)
gbrt_mse = mean_squared_error(train_labels.values.ravel(), train_predictions)
gbrt_rmse = np.sqrt(gbrt_mse)
print("RMSE with Gradient Boosting on  training set is", gbrt_rmse)


boost_scores = cross_val_score(gbrt, train_features, train_labels.values.ravel(), scoring="mean_squared_error", cv=10)
boost_rmse_scores = np.sqrt(-boost_scores)
print('Cross Validation Scores')
display_scores(boost_rmse_scores)

print("R2 cross val= " + str(gbrt.score(train_features, train_labels)))
print("R2 test = " + str(gbrt.score(test_features, test_labels)))


# In[20]:


from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor

param_distribs = {
    'n_estimators': randint(low=1, high=400),
    'learning_rate': [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    'max_features': randint(10, 115),
    'max_depth': randint(low=1, high=4)
}

boost_reg = GradientBoostingRegressor(random_state=42)
rnd_search = RandomizedSearchCV(boost_reg, param_distributions=param_distribs,
                                n_iter=10, cv=10, scoring='mean_squared_error', random_state=42)
rnd_search.fit(train_features, train_labels)

print('Best Params' + str(rnd_search.best_params_))
print('Best Estimator' + str(rnd_search.best_estimator_))
feature_importances = rnd_search.best_estimator_.feature_importances_
#print('Feature importance')
#sorted(zip(feature_importances, attributes), reverse=True)


# In[21]:


final_model = rnd_search.best_estimator_
print('Best Score ' + str(np.sqrt(-rnd_search.best_score_)))

final_predictions = final_model.predict(test_features)
final_mse = mean_squared_error(test_labels, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)

print("R2 test= " + str(final_model.score(test_features, test_labels)))


# In[22]:


# Grid Searching for more granularity
from sklearn.grid_search import GridSearchCV

# Best Params{'max_depth': 3, 'max_features': 51, 'n_estimators': 188, 'learning_rate': 0.4}

param_distribs = [
    {'n_estimators': [150, 175, 200, 225, 250], 'learning_rate': [0.3, 0.4, 0.5], 'max_features': [40, 50, 60]}
]

boost_reg = GradientBoostingRegressor(random_state=42, max_depth=3)
grid_search = GridSearchCV(boost_reg, param_distribs, cv=10, scoring='mean_squared_error')
grid_search.fit(train_features, train_labels)

print('Best Params' + str(grid_search.best_params_))
print('Best Estimator' + str(grid_search.best_estimator_))
feature_importances = grid_search.best_estimator_.feature_importances_
#print('Feature importance')
#sorted(zip(feature_importances, attributes), reverse=True)


final_model = grid_search.best_estimator_
print('RMSE train set  ' + str(np.sqrt(-grid_search.best_score_)))
print('R2 train = ' + str(final_model.score(train_features, train_labels)))

final_test_predictions = final_model.predict(test_features)
final_test_mse = mean_squared_error(test_labels, final_test_predictions)
final_test_rmse = np.sqrt(final_test_mse)
print('RMSE test set  ' + str(final_test_rmse))
print('R2 test= ' + str(final_model.score(test_features, test_labels)))


# In[23]:


# check performance on whole dataset
df_labels = df_trim["Target"].copy()
df_features = full_pipeline.transform(df_trim)

final_predictions = final_model.predict(df_features)
final_mse = mean_squared_error(df_labels, final_predictions)
final_rmse = np.sqrt(final_mse)
print('RMSE whole data set  ' + str(final_rmse))
print('R2 whole data set = ' + str(final_model.score(df_features, df_labels)))


# In[24]:
