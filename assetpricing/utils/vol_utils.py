import scipy as sp

from assetpricing.models.black_scholes import *


def volatility(sigma, args):
    S = args[0]
    K = args[1]
    r = args[2]
    T = args[3]
    q = args[4]
    type = args[5]
    mkt_val = args[6]
    diff = bs_value(S, K, r, T, q, sigma, type) - mkt_val
    return diff


def fvega(sigma, args):
    S = args[0]
    K = args[1]
    r = args[2]
    T = args[3]
    q = args[4]

    vega = bs_vega(S, K, r, T, q, sigma)
    return vega


def implied_volatility(opt_val, spot, strike, expiry, rf, div, option_type, method="Bisection"):
    """
    This function get implied volatility from a call option using either Bisection, Newton or Brent method
    """
    # sigma initial value
    sig_start = np.sqrt(2 * np.pi / expiry) * opt_val / spot
    arglist = (spot, strike, rf, expiry, div, option_type, opt_val)
    argsv = np.array(arglist)

    if method == "Bisection":
        sigma = bisection(volatility, 1e-5, 10.0, argsv, xtol=0.00001)

    # sometimes fails to converge if option value is too small
    elif method == "Newton":
        sigma = sp.optimize.newton(volatility, x0=sig_start, fprime=fvega, args=argsv,
                                   tol=0.00001, maxiter=50, fprime2=None)
    else:
        sigma = sp.optimize.brentq(volatility, 0.00001, 100, maxiter=1000, args=argsv)

    return sigma


def implied_volatility_row(row, method="Bisection"):
    """
    Mimic Implied_Volatilty but using a dataframe row as parameter, in order to increase performance.
    N.B. This function allows to compute all implied vol at once by using df.apply(implied_vol_row)
    instead of having to iterate over the DF
    """

    # Extract values from row
    opt_val = row['lastPrice']
    spot = row['Spot']
    strike = row['strike']
    expiry = row['Expiry']
    rf = row['Risk-Free Rate']
    div = row['Dividend']
    option_type = row['OptionType']

    # sigma initial value
    sig_start = np.sqrt(2 * np.pi / expiry) * opt_val / spot
    arglist = (spot, strike, rf, expiry, div, option_type, opt_val)
    argsv = np.array(arglist)

    if method == "Bisection":
        sigma = bisection(volatility, 1e-5, 10.0, argsv, xtol=0.00001)

    # sometimes fails to converge if option value is too small
    elif method == "Newton":
        sigma = sp.optimize.newton(volatility, x0=sig_start, fprime=fvega, args=argsv,
                                   tol=0.00001, maxiter=50, fprime2=None)
    else:
        sigma = sp.optimize.brentq(volatility, 0.00001, 100, maxiter=1000, args=argsv)

    return sigma


def bisection(func, x1, x2, args, xtol=1e-6, maxIter=100):
    """ Based on Dominic O'kane work on FinancePy.
    Bisection algorithm. You need to supply root brackets x1 and x2. """

    if np.abs(x1-x2) < 1e-10:
        raise Exception("Brackets should not be equal")

    if x1 > x2:
        raise Exception("Bracket x2 should be greater than x1")

    f1 = func(x1, args)
    fmid = func(x2, args)

    if np.abs(f1) < xtol:
        return x1
    elif np.abs(fmid) < xtol:
        return x2

    if f1 * fmid >= 0:
        print("Root not bracketed")
        return None

    for i in range(0, maxIter):

        xmid = (x1 + x2)/2.0
        fmid = func(xmid, args)

        if f1 * fmid < 0:
            x2 = xmid
        else:
            x1 = xmid

        if np.abs(fmid) < xtol:
            return xmid

    print("Bisection exceeded number of iterations", maxIter)
    return None


if __name__ == '__main__':
    r = 0.05
    observed_price = 12.138866898974783
    S1 = 100
    K1 = 110
    q1 = 0.02
    T1 = 0.5

    imp_vol = implied_volatility(observed_price, S1, K1, T1, r, q1, OptionTypes.EUROPEAN_PUT.value)
    print(imp_vol)

    val = bs_value(S1, K1, r, T1, q1, imp_vol, OptionTypes.EUROPEAN_PUT.value)
    print(val)

    S2 = 409.39
    K2 = 195
    t2 = 0.55068
    obs_2 = 1.09
    imp_vol_2 = implied_volatility(obs_2, S2, K2, t2, r, 0, OptionTypes.EUROPEAN_CALL.value)
    print(imp_vol_2)
