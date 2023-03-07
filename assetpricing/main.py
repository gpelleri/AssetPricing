# import financepy as fp
from assetpricing.products.equity.option import *
from scipy.stats import norm
from numpy import np
from utils import *


def montecarlo_sim(S, T, r, q, sigma, steps, N):
    """
    #S = Current stock Price - follows a geometric brownian motion
    #K = Strike Price
    #T = Time to maturity 1 year = 1, 1 months = 1/12
    #r = risk free interest rate
    #q = dividend yield
    # sigma = volatility

    Return
    # [steps,N] Matrix of asset paths
    """
    dt = T / steps
    ST = np.log(S) + np.cumsum(((r - q - sigma ** 2 / 2) * dt +
                                sigma * np.sqrt(dt) *
                                np.random.normal(size=(steps, N))), axis=0)

    return np.exp(ST)


def black_scholes_calc(S0, K, r, T, sigma, q, option_type):
    """This function calculates the value of the European option based on Black-Scholes formula"""
    # 1) determine N(d1) and N(d2)
    d1 = 1 / (sigma * np.sqrt(T)) * (np.log(S0 / K) + (r -q + sigma ** 2 / 2) * T)
    d2 = d1 - sigma * np.sqrt(T)
    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    n_d1 = norm.cdf(-d1)
    n_d2 = norm.cdf(-d2)
    # 2) determine call value
    c = nd1 * S0 * np.exp(-q * T) - nd2 * K * np.exp(-r * T)
    # 3) determine put value
    p = K * np.exp(-r * T) * n_d2 - S0 * np.exp(-q * T) * n_d1
    # 4) define which value to return based on the option_type parameter
    if option_type == 'call':
        return c
    elif option_type == 'put':
        return p
    else:
        print('Wrong option type specified')


def init():
    edf = Stock("EDF", 100, 0.25)
    edf_call = Option("EDF-call", DERIVATIVES, 0.5, edf, 110)
    return edf_call


if __name__ == '__main__':
    """
    S = 100  # stock price
    K = 110  # strike
    T = 1 / 2  # time to maturity
    r = 0.05  # risk-free risk in annual %
    q = 0.02  # annual dividend rate
    sigma = 0.25  # annual volatility in %
    steps = 100  # time steps
    N = 1000  # number of trials

    paths = montecarlo_sim(S, T, r, q, sigma, steps, N)
    """
    # S = 100  # stock price
    # K = 110  # strike
    # T = 1 / 2  # time to maturity
    r = 0.05  # risk-free risk in annual %
    q = 0.02  # annual dividend rate
    # sigma = 0.25  # annual volatility in %
    # steps = 100  # time steps
    call = init()
    print(call.value(100, 110, r, 1 / 2, q, 0.25, "call"))
    #  print(call.bsmValue(call.getUnderlying().getPrice(), call.getStrike(), r, call.getExpiry(), q, 0.25, "call"))

    # S0 = 8.;
    # K = 9.;
    # T = 3 / 12.;
    # r = .01;
    # sigma = .2
    # print(black_scholes_calc(S0, K, r, T, sigma, 0, 'call'))



