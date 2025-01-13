import pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.cross_decomposition import PLSRegression

# Repeat the analysis of Table 3.3 on the spam data discussed in Chapter 1.

# get relative data folder
PATH = pathlib.Path(__file__).resolve().parents[1]
DATA_PATH = PATH.joinpath("data").resolve()

data = pd.read_csv(DATA_PATH.joinpath("prostate.csv"), header=0)
data_train = data.loc[data['train'] == 'T']
data_test = data.loc[data['train'] == 'F']

# print(data)
# print(data_train)
# print(data_test)

x_train = data_train.loc[:, 'lcavol':'pgg45']
x_test = data_test.loc[:, 'lcavol':'pgg45']
y_train = pd.DataFrame(data_train.loc[:, 'lpsa'])
y_test = pd.DataFrame(data_test.loc[:, 'lpsa'])


# a utility function to print results
def displayResults(name, model):
    if name == "Best Subset":
        intercept = model.estimator_.intercept_[0]
        coef = model.estimator_.coef_
    elif name == "PCR":
        lr_tuple = model.steps[1]
        lr_est = lr_tuple[1]
        intercept = lr_est.intercept_[0]
        coef = lr_est.coef_
    elif name == "PLS":
        intercept = model._y_mean[0]
        coef = model.coef_
    else:  # default
        intercept = model.intercept_[0]
        coef = model.coef_

    print("Intercept of final {} model is: {:.3f}".format(name, intercept))
    print("Coefficient of final {} model is: {}".format(name, coef))


# pre-processing
pipeline = Pipeline([('std_scaler', StandardScaler())])
x_train_prepared = pipeline.fit_transform(x_train)
x_test_prepared = pipeline.transform(x_test)

# define various models and their parameter grids
models = [
    {"name": "LS",
    "params": [{'fit_intercept': [True]}],
    "estimator": LinearRegression()
    },
    {"name": "Ridge",
    "params": [
        {'alpha': [0.01, 0.1, 1, 2, 3, 4, 5, 7, 10]}
    ],
    "estimator": Ridge()
    },
    {"name": "Lasso",
    "params": [
        {'alpha': [0.001, 0.01, 0.1, 0.2, 0.5, 1, 5]}
    ],
    "estimator": Lasso()
    },
    {"name": "PLS",
    "params": [
        {'n_components': np.arange(1, 9)}
    ],
    "estimator": PLSRegression(scale=False)
    },
    {"name": "PCR",
    "params": [
        {'pca__n_components': np.arange(1, 9)}
    ],
    "estimator": Pipeline(steps=[('pca', PCA()), ('linear regression', LinearRegression())])
    },
    {"name": "Best Subset",
    "params": [
        {'n_features_to_select': np.arange(1, 9)}
    ],
    "estimator": RFE(LinearRegression())
    }
]

for model in models:
    name = model["name"]
    print("******** Start running model: {} **********".format(name))

    estimator = model["estimator"]
    param_grid = model["params"]
    grid_search = GridSearchCV(estimator, param_grid, cv=10,
                            scoring='neg_mean_squared_error',
                            return_train_score=True)

    grid_search.fit(x_train_prepared, y_train)

    cv_res = grid_search.cv_results_

    for mean_score, params in zip(cv_res["mean_test_score"], cv_res["params"]):
        print("CV results for model {} with parameter {} is {}".format(name, params, np.sqrt(-mean_score)))

    final_model = grid_search.best_estimator_
    final_y_pred = final_model.predict(x_test_prepared)
    final_test_error = mean_squared_error(final_y_pred, y_test)
    displayResults(name, final_model)
    print("Test error of final {} model is : {:.3f}".format(name, final_test_error))
    print("******** End running model: {} **********".format(name))