import numpy as np
import random
import math
import pandas as pd
from types import SimpleNamespace

from scipy.stats import gamma

import statsmodels.api as sm

poss_dist = ['unif', 'norm', 'poiss', 'gamma'] # , 'poiss', 'unif']
poss_fun = ['exp({})', 'log({})', '{}**2', 'sqrt({})', 'sin({})', 'cos({})', '{}'] # '{}**3', '{}**4',


def generate_data(nb_features, size=500, data={}, formula=None, eps_scale=1., distribution=None, seed=random.randint(0, 2**32 - 1)):
    """Complex data generator for cross-sectional tabular dataset.
    The function will draw rancom data from som randomly chosen distributions (normal, poisson or uniform).
    If will also generate the parameters (weights) from a normal distribution whose weights are itself drawn from an random uniform distribution.
    Once, the :math:`\beta` and the random variables :math:`X` are drawn, we may apply a nonlinear transformation fuction on top to :math:`X_{i}`.
    We finally sum the complex transformed variables weithed by their respective weights :math:`\beta`.

    .. math:

        y = \sum_{i}^{p} \beta_i g(X_i) + \varepsilon

    where :math:`g` can be anything of :math:`e^{x}`, :math:`\log(x)`, :math:`x^{2}`, :math:`x^{3}`, :math:`x^{4}`,
    :math:`\sqrt{x}`, :math:`\sin(x)`, :math:`\cos(x)` or :math:`x` (no transformation).

    :param nb_features: Number of random variables to be generated.
    :type nb_features: int
    :param size: Size of the data set to generate, defaults to 500
    :type size: int, optional
    :param data: Data on which we want to stack our randomly generated features, defaults to {}
    :type data: dict, optional
    :param formula: Formula (list of functions) to apply on the original `data` if any, defaults to None
    :type formula: list, optional
    :param eps_scale: Scale parameter for the random noise :math:`\varepsilon`, defaults to 1.
    :type eps_scale: float, optional
    :param seed: Random seed for reproducibility purposes, defaults to 100
    :type seed: int, optional
    :return: Randomly generated data set with complex endogenous varaible `y`.
    :rtype: pandas.DataFrame
    """
    formula = [] if formula is None else formula
    data = {} if data is None else data

    meta = {}

    for i in range(1, nb_features + 1):
        random.seed(seed + (i + 1))
        np.random.seed(seed + (i + 1))
        _dist_i = distribution.lower() if distribution else random.choice(poss_dist)

        _loc_beta = np.random.uniform(low=-2, high=2)
        _scale_beta = np.random.uniform(low=1, high=5)
        _beta = np.random.normal(_loc_beta, _scale_beta, size=1)[0]

        if _dist_i == 'norm':
            # Draw a normal distribution
            # Select the parameters from a uniform distribution (loc and scale)
            _loc = np.random.random() * np.random.randint(10)
            _scale = np.random.uniform(low=1, high=5)
            # And get a sample
            _x = np.random.normal(_loc, _scale, size=size)

            _param = {'str': f'N({_loc:.4f}, {_scale:.4f})', 'loc': _loc, 'scale': _scale}
        elif _dist_i == 'poiss':
            _lambda = np.random.uniform(low=1, high=5)
            _x = np.random.poisson(_lambda, size=size) + 1

            _param = {'str': f'Poisson({_lambda:.4f})', 'lambda': _lambda}
        elif _dist_i == 'gamma':
            _a = np.random.uniform(low=1e-9, high=10)
            _loc = np.random.uniform(low=1e-9, high=10)
            _scale = np.random.uniform(low=1e-9, high=5)
            _x = gamma.rvs(a=_a, loc=_loc, scale=_scale, size=size)

            _param = {'str': f'Gamma({_a:.4f}, {_loc:.4f}, {_scale:.4f})', 'a': _a, 'loc': _loc, 'scale': _scale}
        else:
            _low = np.random.uniform(low=1, high=5)
            _high = _low + np.random.uniform(low=1, high=5)
            _x = np.random.uniform(low=_low, high=_high, size=size)

            _param = {'str': f'Unif({_low:.4f}, {_high:.4f})', 'low': _low, 'high': _high}

        data.update({f'X{i}': _x})

        # Define the formula to compute for the target variable. For random, we do not want to transform with log not sqrt to avoid domain errors
        random.seed(seed + (i + 1))
        if _dist_i == 'norm':
            _transfo = random.choice([f for f in poss_fun if ('log' not in f) and ('sqrt' not in f)])
        else:
            _transfo = random.choice(poss_fun)

        q90 = np.quantile(_x, 0.9)
        _transfo = _transfo.replace('{}', f'(X{i}/{q90})')

        meta.update({f'X{i}': _param, f'beta{i}': _beta, f'f(X{i})': _transfo})
        formula += [f'{_beta} * ' + _transfo]

    # Add the final noise variable
    _eps = np.random.normal(loc=0, scale=eps_scale, size=size)
    data.update({'eps': _eps})
    formula += ['eps']
    meta.update({'eps': f'N(0, {eps_scale:.4f})'})

    form = ' + '.join(formula)
    df = pd.DataFrame(data).eval(f"y = {form}", engine='python')
    df.meta = SimpleNamespace()
    df.meta.distributions = meta
    df.meta.formula = form

    # for c in df.columns + ['ts']:
    #     if (c in meta.keys()) & (c != 'eps'):
    #         df.eval(f"f_{c} = {meta[f'f({c})']}", inplace=True)
    return df


def generate_panel_data(nb_indiv=10, n_cross=5, indiv_sample='random', p=1, q=1, ts_noise=1., eps_scale=1., seed=10):
    """Function to generate synthetic panel data

    :param nb_indiv: Number of individuals in the panel, defaults to 10
    :type nb_indiv: int, optional
    :param indiv_sample: Sample size per individual. Can either be an integer or will be drawn from a uniform distribution, defaults to 'random'
    :type indiv_sample: str, optional
    :param p: Parameter of the AR(p) process, number of lags in the auto-regressive process, defaults to 1
    :type p: int, optional
    :param q: Parameter of the MA(q) process, number of coefficients for the moving-average lag polynomial, defaults to 1
    :type q: int, optional
    :param noise: Standard deviation of noise, value for argument `scale` in the function `sm.tsa.arma_generate_sample`, defaults to 1.
    :type noise: float, optional
    :param seed: Random seed for data generation, defaults to 10
    :type seed: int, optional
    :return: Data frame with one time series variable (for panel data) and cross-sectional variables
    :rtype: pandas.DataFrame
    """
    np.random.seed(seed)
    _nsample = indiv_sample if type(indiv_sample) == int else random.randint(100, 500)
    d_d = []
    arma_params = {}
    for i in range(1, nb_indiv + 1):
        arparams = np.random.uniform(-.2, 0.5, p) if p > 0 else []
        maparams = np.random.uniform(.2, 1., q) if q > 0 else []
        ar = np.r_[1, -arparams]  # add zero-lag and negate
        ma = np.r_[1, maparams]  # add zero-lag
        y = sm.tsa.arma_generate_sample(ar=ar, ma=ma, nsample=_nsample, scale=ts_noise)
        tst = pd.date_range("1980-1-1", freq="d", periods=_nsample)
        d_d += [pd.DataFrame({'ts': y, 'id': f'U{i}', 'timestamp': tst})]
        arma_params.update({f'U{i}': {'AR': arparams, 'MA': maparams}})

    # Stack all individual's data together to get on data set
    _df_ts = pd.concat(d_d).reset_index(drop=True)

    # Get the nonlinear transformation of the time series variable
    random.seed(seed)
    # _transfo = random.choice([f for f in poss_fun if ('log' not in f) and ('sqrt' not in f) and ('sin' not in f) and ('cos' not in f)])
    _transfo = 'cos({})'
    # Generate the weight for the series in the model
    _loc_beta = np.random.uniform(low=-5, high=5, size=1)[0]
    _scale_beta = np.random.uniform(low=1, high=5, size=1)[0]
    _beta = np.random.normal(_loc_beta, _scale_beta, size=1)[0]
    if ('exp' in _transfo) or ('**' in _transfo):
        # If we have exponential/quadratic transformation, we scale the data by the 90% quatile to avoid data explosion
        q90 = np.quantile(_df_ts.ts.values, 0.9)
        _transfo = _transfo.replace('{}', f'(ts/{q90})')
    else:
        _transfo = _transfo.replace('{}', 'ts')

    _data = {
        'ts': _df_ts.ts.values,
        'id': _df_ts.id.values,
        'timestamp': _df_ts.timestamp.values,
    }
    _df = generate_data(n_cross, data=_data, formula=[f'{_beta} * ' + _transfo],
                        size=_df_ts.shape[0], eps_scale=eps_scale, seed=seed+500)
    _df.meta.indiv_sample_size = _nsample
    _df.meta.ARMA = arma_params
    return _df

def generate_cluster_data(nb_indiv=10, n_vars=5, indiv_sample='random',
                          distribution_indiv='gamma', distribution_covariates='gamma',
                          eps_scale=1., seed=random.randint(0, 2**32 - 10000)):
    """Function to generate synthetic panel data

    :param nb_indiv: Number of individuals in the panel, defaults to 10
    :type nb_indiv: int, optional
    :param indiv_sample: Sample size per individual. Can either be an integer or will be drawn from a uniform distribution, defaults to 'random'
    :type indiv_sample: str, optional
    :param noise: Standard deviation of noise, value for argument `scale` in the function `sm.tsa.arma_generate_sample`, defaults to 1.
    :type noise: float, optional
    :param seed: Random seed for data generation, defaults to 10
    :type seed: int, optional
    :return: Data frame with one time series variable (for panel data) and cross-sectional variables
    :rtype: pandas.DataFrame
    """
    np.random.seed(seed)
    _nsample = indiv_sample if type(indiv_sample) == int else random.randint(100, 500)
    d_d = []
    user_params = {}
    for i in range(1, nb_indiv + 1):
        u_df = generate_data(nb_features=1, distribution=distribution_indiv, size=_nsample, seed=seed+i)
        params = u_df.meta.distributions['X1']
        v = u_df.X1
        d_d += [pd.DataFrame({'V1': v, 'id': f'U{i}'})]
        user_params.update({f'U{i}': {'a': params['a'], 'loc': params['loc'], 'scale': params['scale']}})

    # Stack all individual's data together to get on data set
    _df_ts = pd.concat(d_d).reset_index(drop=True)

    # Get the nonlinear transformation of the time series variable
    random.seed(seed)
    # _transfo = random.choice([f for f in poss_fun if ('log' not in f) and ('sqrt' not in f) and ('sin' not in f) and ('cos' not in f)])
    _transfo = 'cos({})'
    # Generate the weight for the series in the model
    _loc_beta = np.random.uniform(low=-5, high=5, size=1)[0]
    _scale_beta = np.random.uniform(low=1, high=5, size=1)[0]
    _beta = np.random.normal(_loc_beta, _scale_beta, size=1)[0]
    if ('exp' in _transfo) or ('**' in _transfo):
        # If we have exponential/quadratic transformation, we scale the data by the 90% quatile to avoid data explosion
        q90 = np.quantile(_df_ts.V1.values, 0.9)
        _transfo = _transfo.replace('{}', f'(V1/{q90})')
    else:
        _transfo = _transfo.replace('{}', 'V1')

    _data = {
        'V1': _df_ts.V1.values,
        'id': _df_ts.id.values,
    }
    _df = generate_data(n_vars, data=_data, formula=[f'{_beta} * ' + _transfo],
                        size=_df_ts.shape[0], eps_scale=eps_scale,
                        distribution=distribution_covariates, seed=seed+10000)
    _df.meta.indiv_sample_size = _nsample
    _df.meta.Users = user_params
    return _df
