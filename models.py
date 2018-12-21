import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


import matplotlib.pyplot as plt
plt.style.use('seaborn')

model_dict={'lasso':Lasso,
            'ridge':Ridge,
            'knr': KNeighborsRegressor,
            'rfr': RandomForestRegressor}

def mean_absolute_ratio_error(y_true, y_pred, epsilon=10):
    return np.mean(np.abs((y_true - y_pred) / max(y_true,epsilon)))

def w_mape(y_true, y_pred):
    return (100/np.mean(y_true)) * mean_absolute_error(y_true=y_true, y_pred=y_pred)


# building features and targets
def prep_data_for_modelisation(data):
    features = data.drop(['TOTALSQUAWKCOST', 'TOTALSQUAWKREVENUE',
                          'WA_TOTALSQUAWKCOST', 'WA_TOTALSQUAWKREVENUE',
                          'WA_TOTALSQUAWKESTIMATEDCOST', 'WA_TOTALSQUAWKESTIMATEDREVENUE'], axis=1)
    cost = data['TOTALSQUAWKCOST']
    wa_cost = data['WA_TOTALSQUAWKCOST']
    wa_estimated_cost = data['WA_TOTALSQUAWKESTIMATEDCOST']
    return features, cost, wa_cost, wa_estimated_cost


def get_low_info_features(data):
    ratio_of_zero = (d==0).sum()/len(d)
    low_info_features = ratio_of_zero[ratio_of_zero>0.8].index
    return low_info_features


def select_lasso_k_best(data, target, k=20):
    #alpha not to high to select enough features
    pg = {'alpha' : np.round(np.logspace(start=0, stop = 2, num = 20), 4)}
    gs = cross_validate_model(data=data,target=target, refit=True, silent=True,
                              model='lasso', param_grid=pg, scale=True)
    importances = np.abs(gs.best_estimator_.coef_)
    k_=min(k, np.count_nonzero(importances))
    indices = np.argsort(importances)[-k_:]
    return data.columns[indices]


def select_features_for_linear_model(data, target):
    d = data.drop("aircraft_model", axis=1, errors='ignore')
    lif = get_low_info_features(d)
    d.drop(lif, axis=1, inplace=True)

    am=pd.get_dummies(data.aircraft_model, prefix='aircraft_').drop('aircraft_0', axis=1, errors='ignore')
    d_=pd.concat([d, m], axis=1)

    best_feat = select_lasso_k_best(d_, target)
    return best_feat


def cross_validate_model(data, target, model, param_grid, verbose=0, silent=False,
                            scale=False, refit=False,
                            scoring=make_scorer(w_mape, greater_is_better=False)):
    if scale:
        # Standardize features to have comparable coefs
        data = StandardScaler().fit_transform(data)

    estimator = model_dict[model]()
    gs = GridSearchCV(estimator=estimator, param_grid=param_grid,
                      cv=5, n_jobs=-1,
                      return_train_score=True, verbose=verbose, refit=refit,
                      scoring=scoring)

    gs.fit(data,target)

    if not silent:
        i = gs.best_index_
        res = [gs.cv_results_['mean_train_score'][i], gs.cv_results_['std_train_score'][i],
               gs.cv_results_['mean_test_score'][i] , gs.cv_results_['std_test_score'][i]]

        print('############# {} ################'.format(model))
        print('Best parameters : {}'.format(gs.cv_results_['params'][i]))
        print('Scores \nTrain: {} % (+/- {})\nTest: {} % (+/- {})\n'.format(res[0], res[1]*2, res[2], res[3]*2))

    return gs


def plot_importances(features, importances):
    indices = np.argsort(importances)
    plt.figure(figsize=(15,20))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)),features[indices])
    plt.xlabel('Relative Importance')
    plt.show()


def get_result(data):
    scores={}
    wa_scores={}
    feat, cost, wa_cost, wa_estimated_cost = prep_data_for_modelisation(data)
    best_feat_cost = feat[select_lasso_k_best(feat, cost)]
    best_feat_wa_cost = feat[select_lasso_k_best(feat, wa_cost)]

    scores['cost_baseline']=w_mape(y_pred=feat.TOTALSQUAWKESTIMATEDCOST, y_true=cost)
    wa_scores['cost_baseline']=w_mape(y_pred=wa_estimated_cost, y_true=wa_cost)


    pg = {'alpha' : np.round(np.logspace(start=-2, stop = 5, num = 20), 4)}
    # cost Lasso Model
    gs = cross_validate_model(data=feat,target=cost, model='lasso', param_grid=pg, scale=True, silent=True)
    scores['Lasso']=gs.cv_results_['mean_test_score'][gs.best_index_]
    # wa cost Lasso Model
    gs = cross_validate_model(data=feat,target=wa_cost, model='lasso', param_grid=pg, scale=True, silent=True)
    wa_scores['Lasso']=gs.cv_results_['mean_test_score'][gs.best_index_]
    # cost Lasso Model with feature selection
    gs = cross_validate_model(data=best_feat_cost,target=cost, model='lasso', param_grid=pg, scale=True, silent=True)
    scores['Lasso with feature selection']=gs.cv_results_['mean_test_score'][gs.best_index_]
    # wa cost Lasso Model with feature selection
    gs = cross_validate_model(data=best_feat_wa_cost,target=wa_cost, model='lasso', param_grid=pg, scale=True, silent=True)
    wa_scores['Lasso with feature selection']=gs.cv_results_['mean_test_score'][gs.best_index_]


    # cost Ridge Model
    gs = cross_validate_model(data=feat,target=cost, model='ridge', param_grid=pg, scale=True, silent=True)
    scores['Ridge']=gs.cv_results_['mean_test_score'][gs.best_index_]
    # wa cost Ridge Model
    gs = cross_validate_model(data=feat,target=wa_cost, model='ridge', param_grid=pg, scale=True, silent=True)
    wa_scores['Ridge']=gs.cv_results_['mean_test_score'][gs.best_index_]
    # cost Ridge Model with feature selection
    gs = cross_validate_model(data=best_feat_cost,target=cost, model='ridge', param_grid=pg, scale=True, silent=True)
    scores['Ridge with feature selection']=gs.cv_results_['mean_test_score'][gs.best_index_]
    # wa cost Ridge Model with feature selection
    gs = cross_validate_model(data=best_feat_wa_cost,target=wa_cost, model='ridge', param_grid=pg, scale=True, silent=True)
    wa_scores['Ridge with feature selection']=gs.cv_results_['mean_test_score'][gs.best_index_]

    return scores, wa_scores



def get_time_result(data):
    scores={}
    feat, _, _, _= prep_data_for_modelisation(data)
    target = feat['length_of_time_in_days']
    features = feat.drop("length_of_time_in_days", axis=1)

    best_feat = feat[select_lasso_k_best(features, target)]


    pg = {'alpha' : np.round(np.logspace(start=-2, stop = 5, num = 20), 4)}
    # time Lasso Model
    gs = cross_validate_model(data=features,target=target, model='lasso', param_grid=pg, scale=True, silent=True, scoring=None)
    scores['Lasso']=gs.cv_results_['mean_test_score'][gs.best_index_]
    # time Lasso Model with feature selection
    gs = cross_validate_model(data=best_feat,target=target, model='lasso', param_grid=pg, scale=True, silent=True, scoring=None)
    scores['Lasso with feature selection']=gs.cv_results_['mean_test_score'][gs.best_index_]

    # time Ridge Model
    gs = cross_validate_model(data=features,target=target, model='ridge', param_grid=pg, scale=True, silent=True, scoring=None)
    scores['Ridge']=gs.cv_results_['mean_test_score'][gs.best_index_]
    # time Ridge Model with feature selection
    gs = cross_validate_model(data=best_feat,target=target, model='ridge', param_grid=pg, scale=True, silent=True, scoring=None)
    scores['Ridge with feature selection']=gs.cv_results_['mean_test_score'][gs.best_index_]

    return scores
