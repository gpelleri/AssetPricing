import numpy as np
import scipy as sp
from assetpricing.models.black_scholes import bs_value, bs_vega
from assetpricing.products.equity.option import Option
from assetpricing.utils.types import *


def implied_volatility_NR(O, S, K, T, r, q, option_type, tol=0.0001,
                       max_iterations=100):
    """
    This function get implied volatilty from a call option using Newton Raphson method

    :param O: Observed option price
    :param S: Asset price
    :param K: Strike Price
    :param T: Time to Maturity
    :param r: risk-free rate
    :param q: dividend yield
    :param tol: error tolerance in result
    :param max_iterations: max iterations to update vol
    :return: implied volatility in percent
    """
    # sigma initial value
    # TODO : better star version is sigma = sqrt(2 pi /T) * C/S
    sigma = np.sqrt(2 * np.pi / T) * O / S

    for i in range(max_iterations):

        ### calculate difference between blackscholes price and market price with
        ### iteratively updated volality estimate
        diff = bs_value(S, K, r, T, q, sigma, option_type) - O

        ### break if difference is less than specified tolerance level
        if abs(diff) < tol:
            print(f'found on {i}th iteration')
            print(f'difference is equal to {diff}')
            break

        ### use newton rapshon to update the estimate
        sigma = sigma - diff / bs_vega(S, K, r, T, q, sigma)

    return sigma

#
# def implied_volatility(O, S, K, T, r, q, option_type, method, tol=0.0001,
#                        max_iterations=100):
#     """
#     This function get implied volatilty associated to a given option price
#
#     :param O: Observed option price
#     :param S: Asset price
#     :param K: Strike Price
#     :param T: Time to Maturity
#     :param r: risk-free rate
#     :param q: dividend yield
#     :param option_type:
#     :param method: allows to use either newton Raphson
#     :param tol: error tolerance in result
#     :param max_iterations: max iterations to update vol
#     :return: implied volatility in percent
#     """
#     # sigma initial value
#     if method == "Newton":
#         sigma = np.sqrt(2 * np.pi / T) * O / S
#
#         for i in range(max_iterations):
#
#             ### calculate difference between blackscholes price and market price with
#             ### iteratively updated volality estimate
#             diff = bs_value(S, K, r, T, q, sigma, option_type) - O
#
#             ### break if difference is less than specified tolerance level
#             if abs(diff) < tol:
#                 print(f'found on {i}th iteration')
#                 print(f'difference is equal to {diff}')
#                 break
#
#             ### use newton rapshon to update the estimate
#             sigma = sigma - diff / bs_vega(S, K, r, T, q, sigma)
#     else:
#         theo_value = bs_value(S, K, r, T, q, sigma, option_type)
#         if option_type == OptionTypes.EUROPEAN_CALL:
#           res = sp.minimize_scalar(abs(O,), bounds=(0.01, 6), method='bounded')
#
#         elif option_type == OptionTypes.EUROPEAN_PUT:
#             res = sp.minimize_scalar(O, bounds=(0.01, 6),
#                                   method='bounded')
#         else:
#             raise Exception("Wrong option type")
#         sigma = res.x
#
#     return sigma