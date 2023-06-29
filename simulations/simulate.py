import math
import warnings
import datetime
import pandas as pd
import numpy as np
import pycof as pc

from tqdm import tqdm

import statsmodels.api as sm
from scipy import stats
from scipy.stats import gamma

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from data_generators import poss_dist

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default="/home/florian/StatisticallyEnhancedLearning/.data/")
parser.add_argument("--iterations_per_comb", type=int, default=20, help="Number of iteration using a fixed set of parameters.")
parser.add_argument("--auto_shutdown", action="store_true", help="Shut down instance once simulation is done.")
parser.add_argument("--save_every", type=int, default=5, help="Save grid with results every X iterations.")
parser.add_argument("--var_sel", type=str, default='y', help="Variable for SEL features (moment and MLE). If variable is y (default), then we hide the variable V1.")
parser.add_argument("--distribution_sel", type=str, default='gamma', choices=poss_dist, help="Distribution of the SEL variable at individual/cluster level.")
parser.add_argument("--distribution_covariates", type=str, default='gamma', choices=poss_dist, help="Distribution of the other corss-sectional variables.")
parser.add_argument("--max_features", type=str, default='sqrt', help="Max features argument for Random Forest.")
parser.add_argument("--rf_ntrees", type=int, default=500, help="Number of trees for the Random Forest.")
parser.add_argument("--n_jobs", type=int, default=-1, help="Number of CPUs to use for Random Forest")
parser.add_argument("--debug", action="store_true", help="Run debug mode to display details.")
parser.add_argument("--form_known", action="store_true", help="Assume form of Cauchy parameters is known")
parser.add_argument("--tree_method", default='gpu_hist', help="Tree method for XGBoost model.")
args = parser.parse_args()

print('##### SEL SIMULATIONS WITH SYNTHETIC DATA #####')


_now = datetime.datetime.now()
os.makedirs(args.save_path, exist_ok=True)
path_res = os.path.join(args.save_path, f"simulation_{'v3' if args.form_known else 'v4'}_results_{_now.strftime('%y%m%d_%H%M%S')}.parquet")

print('')
print('Running the SEL simulations for the variable', args.var_sel, 'with a', args.distribution_sel, 'distribution.')
print('Other covariates will follow a', args.distribution_covariates, 'distribution.')
print('')
print('Simulation parameters are:')
print('    • Iterations per combination:', args.iterations_per_comb)
print('    • Random Forest max features:', args.max_features)
print('    • Random Forest number of trees:', args.rf_ntrees)
print('    • Random Forest number of jobs:', args.n_jobs)
print('')
print('Script parameters are:')
print('    • Results will be saved every {} iterations at: {}'.format(args.save_every, path_res))
print('    • Instance will shut down once simulation is done' if args.auto_shutdown else '')
print('')

# Filter unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Global hyper_parameters for Random Forest

# Hyper-parameters. We will iterate on all of them in the future.
grid_cols = ['indiv_sample', 'nb_indiv', 'n_cols', 'n', 'q', 'ts_noise', 'cs_noise']
grid_all_comb = [(indiv_sample, nb_indiv, n_cols, n, q, ts_noise, cs_noise)
                 for indiv_sample in [500] * args.iterations_per_comb  # n
                 for nb_indiv in [400]  # m
                 for n_cols in list(np.arange(5, 50, 5))  # p
                 for n in [500]
                 for q in [1]
                 for ts_noise in [1.]
                 for cs_noise in [1.]
                 ]

iteration_grid = pd.DataFrame(grid_all_comb, columns=grid_cols).sample(frac=1).reset_index(drop=True)
# Initialize some columns to fill in later
iteration_grid['rmse_ols_vanilla'] = np.nan
iteration_grid['rmse_ols_moments'] = np.nan
iteration_grid['rmse_ols_sel'] = np.nan
iteration_grid['rmse_rf_vanilla'] = np.nan
iteration_grid['rmse_rf_moments'] = np.nan
iteration_grid['rmse_rf_sel'] = np.nan
iteration_grid['rmse_xgb_vanilla'] = np.nan
iteration_grid['rmse_xgb_moments'] = np.nan
iteration_grid['rmse_xgb_sel'] = np.nan
# Glance for features of importance
iteration_grid['formula_data'] = ''
iteration_grid['formula_ols_vanilla'] = ''
iteration_grid['formula_ols_moments'] = ''
iteration_grid['formula_ols_sel'] = ''
iteration_grid['formula_rf_vanilla'] = ''
iteration_grid['formula_rf_moments'] = ''
iteration_grid['formula_rf_sel'] = ''
iteration_grid['indiv_sample_size'] = np.nan

iteration_grid['sel_params_real'] = np.nan
iteration_grid['sel_params_est'] = ''
# Add information of which is the SEL variable
iteration_grid['sel_var'] = args.var_sel

iteration_grid.to_parquet(path_res)

_it = 1

print(datetime.datetime.now().replace(microsecond=0), '--> Running simulation')
print('')

max_it = iteration_grid.shape[0]

disp = list if args.debug else tqdm

for index, row in disp(iteration_grid.iterrows()):
    pc.verbose_display(f'Generating data ({index}/{max_it})....................................', end='\r', verbose=args.debug)
    df = pd.DataFrame()
    lambdas_df = pd.DataFrame()

    # Generate the p observable variables
    d = {}
    for i in range(row.n_cols):
        d.update({f'X{i}': np.random.normal(5.544, 2.7841, size=row.indiv_sample)})

    # Generate the params for the Cauchy distribution
    d.update({'alpha': np.random.normal(5., 2., size=row.indiv_sample)})
    d.update({'beta': np.random.normal(3., 1., size=row.indiv_sample)})
    d.update({'scale': [1.] * row.indiv_sample})

    df = pd.DataFrame(d)

    df.alpha = df.alpha.apply(abs)
    df.beta = df.beta.apply(abs)
    df.scale = df.scale.apply(abs)

    pc.verbose_display(f'Estimating SEL parameters ({index}/{max_it})....................................', end='\r', verbose=args.debug)

    hidden_series = {}
    est_alphas = []
    est_betas = []
    est_scales = []
    y_mean = []
    for _i, _row in df.iterrows():
        _d = stats.cauchy.rvs(loc=_row.beta, scale=_row.scale, size=row.nb_indiv)
        # _d = stats.gamma.rvs(row.alpha, loc=row.beta, scale=row.scale, size=m)
        _b, _s = stats.cauchy.fit(_d)
        # _a, _b, _s = stats.gamma.fit(_d)
        # est_alphas += [_a]
        est_betas += [_b]
        est_scales += [_s]
        y_mean += [_d.mean()]
        hidden_series.update({f'Y{_i}': _d})

    est_params = pd.DataFrame({
                            'beta_hat': est_betas,
                            'scale_hat': est_scales,
                            'y_mean': y_mean})

    iteration_grid.loc[index, 'indiv_sample_size'] = row.indiv_sample

    pc.verbose_display(f'Merging sets with SEL variables ({index}/{max_it})....................................', end='\r', verbose=args.debug)

    df_full = df.join(est_params)

    pc.verbose_display(f'Defining target variable ({index}/{max_it})...........................................', end='\r', verbose=args.debug)

    df_full['b**2+l'] = (df_full['beta']**2) + df_full['scale']  # ) * df_full['scale']

    if args.form_known:
        df_full['hat(b**2+l)'] = (df_full['beta_hat']**2) + df_full['scale_hat']
        sel_cols = ['hat(b**2+l)']
    else:
        df_full['hat(b)'] = df_full['beta_hat']
        df_full['hat(l)'] = df_full['scale_hat']
        sel_cols = ['hat(b)', 'hat(l)']

    betas = np.random.uniform(-5, 5, size=row.n_cols)

    # Computing the target variable
    df_full['Z'] = 0
    for i in range(row.n_cols):
        if i == 0:
            df_full['Z'] += betas[i] * (df_full[f'X{i}'])
        else:
            df_full['Z'] += betas[i] * df_full[f'X{i}']

    df_full['Z'] += np.abs(betas).max() * df_full['b**2+l']
    df_full['Z'] /= 10  # df_full['Z'].mean()

    exog_cols = [f'X{i}' for i in range(row.n_cols)]
    mmt_cols = ['y_mean']

    train, test = train_test_split(df_full, test_size=0.3)

    # ### OLS ###
    # try:
    pc.verbose_display(f'Fitting Vanilla OLS ({index}/{max_it})....................................', end='\r', verbose=args.debug)
    # We fit a first 'vanilla' OLS, with observed variables only
    ols_vanilla = sm.OLS(train['Z'], sm.add_constant(train[exog_cols])).fit()
    Y_test_pred = ols_vanilla.predict(sm.add_constant(test[exog_cols]))

    # We compute the accuracy (RMSE) of the model and also extract the estimated weights (for deep dives on feature importance)
    iteration_grid.loc[index, 'rmse_ols_vanilla'] = mean_squared_error(test['Z'], Y_test_pred, squared=False)
    # iteration_grid.loc[index, 'formula_ols_vanilla'] = ' + '.join([f'{i[1]} * {i[0]}' for i in ols_vanilla.params.items()])

    pc.verbose_display(f'Fitting Moments OLS ({index}/{max_it})....................................', end='\r', verbose=args.debug)
    # We then fit an OLS based on observed features + extracted moments
    ols_mmnt = sm.OLS(train['Z'], sm.add_constant(train[exog_cols + mmt_cols])).fit()
    Y_test_pred_mmnt = ols_mmnt.predict(sm.add_constant(test[exog_cols + mmt_cols]))

    # We compute the accuracy (RMSE) of the model and also extract the estimated weights (for deep dives on feature importance)
    iteration_grid.loc[index, 'rmse_ols_moments'] = mean_squared_error(test['Z'], Y_test_pred_mmnt, squared=False)
    # iteration_grid.loc[index, 'formula_ols_moments'] = ' + '.join([f'{i[1]} * {i[0]}' for i in ols_mmnt.params.items()])

    pc.verbose_display(f'Fitting SEL OLS ({index}/{max_it})....................................', end='\r', verbose=args.debug)
    # We finally fit an OLS based on observed features + SEL variables (estimated parameters of ARIMA)
    ols_sel = sm.OLS(train['Z'], sm.add_constant(train[exog_cols + sel_cols])).fit()
    Y_test_pred_sel = ols_sel.predict(sm.add_constant(test[exog_cols + sel_cols]))

    # We compute the accuracy (RMSE) of the model and also extract the estimated weights (for deep dives on feature importance)
    iteration_grid.loc[index, 'rmse_ols_sel'] = mean_squared_error(test['Z'], Y_test_pred_sel, squared=False)
    # iteration_grid.loc[index, 'formula_ols_sel'] = ' + '.join([f'{i[1]} * {i[0]}' for i in ols_sel.params.items()])
    # except Exception:
    #     continue

    # ### RANDOM FOREST ###
    # try:
    pc.verbose_display(f'Fitting Vanilla RF ({index}/{max_it})....................................', end='\r', verbose=args.debug)
    # We now fit a Vanilla Random Forest based on observed variables
    rf_vanilla = RandomForestRegressor(max_features='sqrt', n_estimators=args.rf_ntrees, n_jobs=args.n_jobs)
    fitted_vanilla = rf_vanilla.fit(train[exog_cols], train['Z'])
    Y_test_rf = rf_vanilla.predict(test[exog_cols])

    # Get features of importance
    importances = fitted_vanilla.feature_importances_

    iteration_grid.loc[index, 'rmse_rf_vanilla'] = mean_squared_error(test['Z'], Y_test_rf, squared=False)
    # iteration_grid.loc[index, 'formula_rf_vanilla'] = ' + '.join([f'{importances[i]} * {X.columns[i]}' for i in range(len(X.columns))])

    pc.verbose_display(f'Fitting Moments RF ({index}/{max_it})....................................', end='\r', verbose=args.debug)
    # We now then fit a Random Forest based on observed variables + moments
    rf_mmnt = RandomForestRegressor(max_features='sqrt', n_estimators=args.rf_ntrees, n_jobs=args.n_jobs)
    fitted_mmnt = rf_mmnt.fit(train[exog_cols + mmt_cols], train['Z'])
    Y_test_rf_mmnt = rf_mmnt.predict(test[exog_cols + mmt_cols])

    # Get features of importance
    importances_mmnt = fitted_mmnt.feature_importances_

    iteration_grid.loc[index, 'rmse_rf_moments'] = mean_squared_error(test['Z'], Y_test_rf_mmnt, squared=False)
    # iteration_grid.loc[index, 'formula_rf_moments'] = ' + '.join([f'{importances_mmnt[i]} * {X_mmnt.columns[i]}' for i in range(len(X_mmnt.columns))])

    pc.verbose_display(f'Fitting SEL RF ({index}/{max_it})....................................', end='\r', verbose=args.debug)
    # We finally fit a Random Forest based on observed variables + SEL features
    rf_sel = RandomForestRegressor(max_features='sqrt', n_estimators=args.rf_ntrees, n_jobs=args.n_jobs)
    fitted_sel = rf_sel.fit(train[exog_cols + sel_cols], train['Z'])
    Y_test_rf_sel = rf_sel.predict(test[exog_cols + sel_cols])

    # Get features of importance
    importances_sel = fitted_sel.feature_importances_

    iteration_grid.loc[index, 'rmse_rf_sel'] = mean_squared_error(test['Z'], Y_test_rf_sel, squared=False)

    # ### XGBoost ###
    # try:
    pc.verbose_display(f'Fitting Vanilla XGB ({index}/{max_it})....................................', end='\r', verbose=args.debug)
    # We now fit a Vanilla XGBoost based on observed variables
    xgb_vanilla = XGBRegressor(n_estimators=args.rf_ntrees, n_jobs=args.n_jobs, tree_method=args.tree_method)
    fitted_vanilla = xgb_vanilla.fit(train[exog_cols], train['Z'])
    Y_test_xgb = xgb_vanilla.predict(test[exog_cols])

    # Get features of importance
    importances = fitted_vanilla.feature_importances_

    iteration_grid.loc[index, 'rmse_xgb_vanilla'] = mean_squared_error(test['Z'], Y_test_xgb, squared=False)
    # iteration_grid.loc[index, 'formula_rf_vanilla'] = ' + '.join([f'{importances[i]} * {X.columns[i]}' for i in range(len(X.columns))])

    pc.verbose_display(f'Fitting Moments XGB ({index}/{max_it})....................................', end='\r', verbose=args.debug)
    # We now then fit a XGBoost based on observed variables + moments
    xgb_mmnt = XGBRegressor(n_estimators=args.rf_ntrees, n_jobs=args.n_jobs, tree_method=args.tree_method)
    fitted_mmnt = xgb_mmnt.fit(train[exog_cols + mmt_cols], train['Z'])
    Y_test_xgb_mmnt = xgb_mmnt.predict(test[exog_cols + mmt_cols])

    # Get features of importance
    importances_mmnt = fitted_mmnt.feature_importances_

    iteration_grid.loc[index, 'rmse_xgb_moments'] = mean_squared_error(test['Z'], Y_test_xgb_mmnt, squared=False)
    # iteration_grid.loc[index, 'formula_rf_moments'] = ' + '.join([f'{importances_mmnt[i]} * {X_mmnt.columns[i]}' for i in range(len(X_mmnt.columns))])

    pc.verbose_display(f'Fitting SEL XGB ({index}/{max_it})....................................', end='\r', verbose=args.debug)
    # We finally fit a Random Forest based on observed variables + SEL features
    xgb_sel = XGBRegressor(n_estimators=args.rf_ntrees, n_jobs=args.n_jobs, tree_method=args.tree_method)
    fitted_sel = xgb_sel.fit(train[exog_cols + sel_cols], train['Z'])
    Y_test_xgb_sel = xgb_sel.predict(test[exog_cols + sel_cols])

    # Get features of importance
    importances_sel = fitted_sel.feature_importances_

    iteration_grid.loc[index, 'rmse_xgb_sel'] = mean_squared_error(test['Z'], Y_test_xgb_sel, squared=False)
    # iteration_grid.loc[index, 'formula_rf_sel'] = ' + '.join([f'{importances_sel[i]} * {X_sel.columns[i]}' for i in range(len(X_sel.columns))])
    # except Exception:
    #     continue

    # Save the iterations information (o get almost live results)
    if _it == args.save_every:
        iteration_grid.to_parquet(path_res)
        _it = 0

    _it += 1

print('')
print(datetime.datetime.now().replace(microsecond=0), '--> Done!')

if args.auto_shutdown:
    os.system('sudo shutdown now')
