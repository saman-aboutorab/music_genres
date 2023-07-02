# Music genres
:::

::: {.cell .markdown}
## Introduction
:::

::: {.cell .markdown id="vMXN-3IIJVbS"}
\"genre\" has been converted to a binary feature where 1 indicates a
rock song, and 0 represents other genres.
:::

::: {.cell .markdown}
![music](vertopal_709426f33f6d49bbb2b32c82d089e4ac/ecad38c2c3973a6df483e839cab040d8023c83d0.jpg)
:::

::: {.cell .code execution_count="20" id="FXQ1Xcx9EEio"}
``` python
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score, KFold, train_test_split

from sklearn.linear_model import Ridge

# Import modules
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.linear_model import Lasso

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV
```
:::

::: {.cell .code execution_count="12" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="xVXBTdBFEI73" outputId="aa8a1878-7cab-4491-94c3-b33a350e1f59"}
``` python
music_df = pd.read_csv('music_clean.csv')
print(music_df.head(5))
```

::: {.output .stream .stdout}
       Unnamed: 0  popularity  acousticness  danceability  duration_ms  energy  \
    0       36506        60.0      0.896000         0.726     214547.0   0.177   
    1       37591        63.0      0.003840         0.635     190448.0   0.908   
    2       37658        59.0      0.000075         0.352     456320.0   0.956   
    3       36060        54.0      0.945000         0.488     352280.0   0.326   
    4       35710        55.0      0.245000         0.667     273693.0   0.647   

       instrumentalness  liveness  loudness  speechiness    tempo  valence  genre  
    0          0.000002    0.1160   -14.824       0.0353   92.934    0.618      1  
    1          0.083400    0.2390    -4.795       0.0563  110.012    0.637      1  
    2          0.020300    0.1250    -3.634       0.1490  122.897    0.228      1  
    3          0.015700    0.1190   -12.020       0.0328  106.063    0.323      1  
    4          0.000297    0.0633    -7.787       0.0487  143.995    0.300      1  
:::
:::

::: {.cell .code execution_count="13" id="SeDs8kHMIRxm"}
``` python
X = music_df.drop('genre', axis=1)
y = music_df[['genre']]
```
:::

::: {.cell .code execution_count="14" id="wKq5zLJDIG-j"}
``` python

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```
:::

::: {.cell .markdown id="__A58WXdGmz5"}
## Pipeline
:::

::: {.cell .code execution_count="15" id="4-nug5a6GVvi"}
``` python
# Instantiate an imputer
imputer = SimpleImputer()

# Instantiate a knn model
knn = KNeighborsClassifier(n_neighbors=3)

# Build steps for the pipeline
steps = [("imputer", imputer),
         ("knn", knn)]
```
:::

::: {.cell .code execution_count="16" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="XZSIihyiGrPj" outputId="4c9db1e7-7400-4ca3-a902-2d5a9cf8700d"}
``` python
steps = [("imputer", imputer),
        ("knn", knn)]

# Create the pipeline
pipeline = Pipeline(steps)

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Print the confusion matrix
print(confusion_matrix(y_test, y_pred))
```

::: {.output .stream .stdout}
    [[89 11]
     [ 3 97]]
:::

::: {.output .stream .stderr}
    /usr/local/lib/python3.10/dist-packages/sklearn/neighbors/_classification.py:215: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      return self._fit(X, y)
:::
:::

::: {.cell .code execution_count="17" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="goAXpzryEtsv" outputId="62dbc51a-db30-44f4-b0e6-b6507037d2d9"}
``` python
# Create X and y
X = music_df.drop('popularity', axis=1).values
y = music_df['popularity'].values

# Instantiate a ridge model
ridge = Ridge(alpha=0.2)

# Instantiate Kfold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
scores = cross_val_score(ridge, X, y, cv=kf, scoring="neg_mean_squared_error")

# Calculate RMSE
rmse = np.sqrt(-scores)
print("Average RMSE: {}".format(np.mean(rmse)))
print("Standard Deviation of the target array: {}".format(np.std(y)))
```

::: {.output .stream .stdout}
    Average RMSE: 10.033098690539362
    Standard Deviation of the target array: 14.02156909907019
:::
:::

::: {.cell .markdown id="9HlRLbqYF5vj"}
An average RMSE of approximately 8.24 is lower than the standard
deviation of the target variable (song popularity), suggesting the model
is reasonably accurate.
:::

::: {.cell .code execution_count="18" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="ascc3UcHFjJa" outputId="91e00da2-e930-4b23-fe4e-dee0fe3f3074"}
``` python
# Import StandardScaler
from sklearn.preprocessing import StandardScaler

X = music_df.drop('loudness', axis=1).values
y = music_df['loudness'].values

# Create pipeline steps
steps = [("scaler", StandardScaler()),
         ("lasso", Lasso(alpha=0.5))]

# Instantiate the pipeline
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)

# Calculate and print R-squared
print(pipeline.score(X_test, y_test))
```

::: {.output .stream .stdout}
    0.0
:::
:::

::: {.cell .markdown id="XfxQ4N8-ZIjd"}
## Centering and scaling
:::

::: {.cell .code execution_count="23" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":1000}" id="0BQ40SlIIqvm" outputId="1b24f803-4e7a-45f1-d8c6-68a4e764770c"}
``` python
# Build the steps
steps = [("scaler", StandardScaler()),
         ("logreg", LogisticRegression())]
pipeline = Pipeline(steps)

# Create the parameter space
parameters = {"logreg__C": np.linspace(0.001, 1.0, 20)}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=21)

# Instantiate the grid search object
cv = GridSearchCV(pipeline, param_grid=parameters)

print(X_train)

# Fit to the training data
cv.fit(X_train, y_train)
print(cv.best_score_, "\n", cv.best_params_)
```

::: {.output .stream .stdout}
    [[3.95290e+04 7.10000e+01 3.11000e-02 ... 1.03987e+02 7.28000e-01
      1.00000e+00]
     [3.89730e+04 6.40000e+01 6.64000e-01 ... 1.47694e+02 1.16000e-01
      1.00000e+00]
     [4.27270e+04 0.00000e+00 9.56000e-01 ... 1.36270e+02 5.50000e-02
      0.00000e+00]
     ...
     [4.99400e+04 4.80000e+01 1.98000e-01 ... 1.29538e+02 2.00000e-01
      0.00000e+00]
     [3.84300e+04 6.50000e+01 2.87000e-05 ... 1.11132e+02 4.38000e-01
      1.00000e+00]
     [1.57000e+03 5.90000e+01 4.08000e-01 ... 1.15034e+02 4.32000e-01
      0.00000e+00]]
:::

::: {.output .error ename="ValueError" evalue="ignored"}
    ---------------------------------------------------------------------------
    ValueError                                Traceback (most recent call last)
    <ipython-input-23-8eeea2242c4c> in <cell line: 17>()
         15 
         16 # Fit to the training data
    ---> 17 cv.fit(X_train, y_train)
         18 print(cv.best_score_, "\n", cv.best_params_)

    /usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_search.py in fit(self, X, y, groups, **fit_params)
        872                 return results
        873 
    --> 874             self._run_search(evaluate_candidates)
        875 
        876             # multimetric is determined here because in the case of a callable

    /usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_search.py in _run_search(self, evaluate_candidates)
       1386     def _run_search(self, evaluate_candidates):
       1387         """Search all candidates in param_grid"""
    -> 1388         evaluate_candidates(ParameterGrid(self.param_grid))
       1389 
       1390 

    /usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_search.py in evaluate_candidates(candidate_params, cv, more_results)
        849                     )
        850 
    --> 851                 _warn_or_raise_about_fit_failures(out, self.error_score)
        852 
        853                 # For callable self.scoring, the return type is only know after

    /usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_validation.py in _warn_or_raise_about_fit_failures(results, error_score)
        365                 f"Below are more details about the failures:\n{fit_errors_summary}"
        366             )
    --> 367             raise ValueError(all_fits_failed_message)
        368 
        369         else:

    ValueError: 
    All the 100 fits failed.
    It is very likely that your model is misconfigured.
    You can try to debug the error by setting error_score='raise'.

    Below are more details about the failures:
    --------------------------------------------------------------------------------
    100 fits failed with the following error:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_validation.py", line 686, in _fit_and_score
        estimator.fit(X_train, y_train, **fit_params)
      File "/usr/local/lib/python3.10/dist-packages/sklearn/pipeline.py", line 405, in fit
        self._final_estimator.fit(Xt, y, **fit_params_last_step)
      File "/usr/local/lib/python3.10/dist-packages/sklearn/linear_model/_logistic.py", line 1204, in fit
        check_classification_targets(y)
      File "/usr/local/lib/python3.10/dist-packages/sklearn/utils/multiclass.py", line 218, in check_classification_targets
        raise ValueError("Unknown label type: %r" % y_type)
    ValueError: Unknown label type: 'continuous'
:::
:::

::: {.cell .markdown id="OXjUQcXEcPH5"}
## Visualizing regression model performance
:::

::: {.cell .code execution_count="24" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":294}" id="7OJqSgfZKQoG" outputId="ac46b5f3-c943-44d7-a38c-23ee6420c6cc"}
``` python
models = {"Linear Regression": LinearRegression(), "Ridge": Ridge(alpha=0.1), "Lasso": Lasso(alpha=0.1)}
results = []

# Loop through the models' values
for model in models.values():
  kf = KFold(n_splits=6, random_state=42, shuffle=True)

  # Perform cross-validation
  cv_scores = cross_val_score(model, X_train, y_train, cv=kf)

  # Append the results
  results.append(cv_scores)

# Create a box plot of the results
plt.boxplot(results, labels=models.keys())
plt.show()
```

::: {.output .error ename="NameError" evalue="ignored"}
    ---------------------------------------------------------------------------
    NameError                                 Traceback (most recent call last)
    <ipython-input-24-038a9d880ab3> in <cell line: 1>()
    ----> 1 models = {"Linear Regression": LinearRegression(), "Ridge": Ridge(alpha=0.1), "Lasso": Lasso(alpha=0.1)}
          2 results = []
          3 
          4 # Loop through the models' values
          5 for model in models.values():

    NameError: name 'LinearRegression' is not defined
:::
:::

::: {.cell .markdown id="ST4qgRZydblM"}
## Predict
:::

::: {.cell .code execution_count="25" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":276}" id="W61CfNWVcRab" outputId="65d63952-c56e-4427-a360-ebc55a259677"}
``` python
# Import mean_squared_error
from sklearn.metrics import mean_squared_error

for name, model in models.items():

  # Fit the model to the training data
  model.fit(X_train_scaled, y_train)

  # Make predictions on the test set
  y_pred = model.predict(X_test_scaled)

  # Calculate the test_rmse
  test_rmse = mean_squared_error(y_test, y_pred, squared=False)
  print("{} Test Set RMSE: {}".format(name, test_rmse))
```

::: {.output .error ename="NameError" evalue="ignored"}
    ---------------------------------------------------------------------------
    NameError                                 Traceback (most recent call last)
    <ipython-input-25-b65ad35f7998> in <cell line: 4>()
          2 from sklearn.metrics import mean_squared_error
          3 
    ----> 4 for name, model in models.items():
          5 
          6   # Fit the model to the training data

    NameError: name 'models' is not defined
:::
:::

::: {.cell .markdown id="8ogvBxaBdiXc"}
## Visualizing classification model performance
:::

::: {.cell .code execution_count="26" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":312}" id="IlF3oQd7dfLt" outputId="af029bf7-3eec-4425-c56f-9bc779bf399c"}
``` python
# Create models dictionary
models = {"Logistic Regression": LogisticRegression(), "KNN": KNeighborsClassifier(), "Decision Tree Classifier": DecisionTreeClassifier()}
results = []

# Loop through the models' values
for model in models.values():

  # Instantiate a KFold object
  kf = KFold(n_splits=6, random_state=12, shuffle=True)

  # Perform cross-validation
  cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf)
  results.append(cv_results)
plt.boxplot(results, labels=models.keys())
plt.show()
```

::: {.output .error ename="NameError" evalue="ignored"}
    ---------------------------------------------------------------------------
    NameError                                 Traceback (most recent call last)
    <ipython-input-26-f0cde50c0edc> in <cell line: 2>()
          1 # Create models dictionary
    ----> 2 models = {"Logistic Regression": LogisticRegression(), "KNN": KNeighborsClassifier(), "Decision Tree Classifier": DecisionTreeClassifier()}
          3 results = []
          4 
          5 # Loop through the models' values

    NameError: name 'DecisionTreeClassifier' is not defined
:::
:::

::: {.cell .code execution_count="27" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":907}" id="lbtCzryZdkWV" outputId="8a8bff1b-99e7-4124-bb4b-49fd83dd93a7"}
``` python
# Create steps
steps = [("imp_mean", SimpleImputer()),
         ("scaler", StandardScaler()),
         ("logreg", LogisticRegression())]

# Set up pipeline
pipeline = Pipeline(steps)
params = {"logreg__solver": ["newton-cg", "saga", "lbfgs"],
         "logreg__C": np.linspace(0.001, 1.0, 10)}

# Create the GridSearchCV object
tuning = GridSearchCV(pipeline, param_grid=params)
tuning.fit(X_train, y_train)
y_pred = tuning.predict(X_test)

# Compute and print performance
print("Tuned Logistic Regression Parameters: {}, Accuracy: {}".format(tuning.best_params_, tuning.score(X_test, y_test) ))
```

::: {.output .error ename="ValueError" evalue="ignored"}
    ---------------------------------------------------------------------------
    ValueError                                Traceback (most recent call last)
    <ipython-input-27-8a65aa4faafd> in <cell line: 13>()
         11 # Create the GridSearchCV object
         12 tuning = GridSearchCV(pipeline, param_grid=params)
    ---> 13 tuning.fit(X_train, y_train)
         14 y_pred = tuning.predict(X_test)
         15 

    /usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_search.py in fit(self, X, y, groups, **fit_params)
        872                 return results
        873 
    --> 874             self._run_search(evaluate_candidates)
        875 
        876             # multimetric is determined here because in the case of a callable

    /usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_search.py in _run_search(self, evaluate_candidates)
       1386     def _run_search(self, evaluate_candidates):
       1387         """Search all candidates in param_grid"""
    -> 1388         evaluate_candidates(ParameterGrid(self.param_grid))
       1389 
       1390 

    /usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_search.py in evaluate_candidates(candidate_params, cv, more_results)
        849                     )
        850 
    --> 851                 _warn_or_raise_about_fit_failures(out, self.error_score)
        852 
        853                 # For callable self.scoring, the return type is only know after

    /usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_validation.py in _warn_or_raise_about_fit_failures(results, error_score)
        365                 f"Below are more details about the failures:\n{fit_errors_summary}"
        366             )
    --> 367             raise ValueError(all_fits_failed_message)
        368 
        369         else:

    ValueError: 
    All the 150 fits failed.
    It is very likely that your model is misconfigured.
    You can try to debug the error by setting error_score='raise'.

    Below are more details about the failures:
    --------------------------------------------------------------------------------
    150 fits failed with the following error:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_validation.py", line 686, in _fit_and_score
        estimator.fit(X_train, y_train, **fit_params)
      File "/usr/local/lib/python3.10/dist-packages/sklearn/pipeline.py", line 405, in fit
        self._final_estimator.fit(Xt, y, **fit_params_last_step)
      File "/usr/local/lib/python3.10/dist-packages/sklearn/linear_model/_logistic.py", line 1204, in fit
        check_classification_targets(y)
      File "/usr/local/lib/python3.10/dist-packages/sklearn/utils/multiclass.py", line 218, in check_classification_targets
        raise ValueError("Unknown label type: %r" % y_type)
    ValueError: Unknown label type: 'continuous'
:::
:::

::: {.cell .code id="nDsrFbOLdrha"}
``` python
```
:::
