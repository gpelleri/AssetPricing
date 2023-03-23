import numpy as np
from scipy.stats import norm

from assetpricing.models.binom_tree import BinomialTreeOption
from assetpricing.models.crr_tree import CRRTreeOption
from assetpricing.utils.types import OptionTypes, BlackScholesTypes

"""
This class implements Black & Scholes & Merton model.
It also features greek computations methods based on BS analytical
"""


class BlackScholes():
    def __init__(self, implementationType=BlackScholesTypes.DEFAULT, num_step_per_year=52, pu=None, pd=None):
        """
        :param implementationType: Black Scholes implementation : analytical, binomial tree or CRR tree
        :param num_step_per_year: only necessary for Tree implementation - Default if 52
        :param pu: Mandatory for BINOMIAL TREE - Default to None
        :param pd: Mandatory for BINOMIAL TREE - Default to None
        """
        self._implementationType = implementationType
        self._num_step = num_step_per_year
        self._pu = pu
        self._pd = pd

    def value(self,
              spotPrice: float,
              strike_price: float,
              risk_free_rate: float,
              time_to_expiry: float,
              dividendRate: float,
              volatility: float,
              option_type: OptionTypes):

        if option_type == OptionTypes.EUROPEAN_CALL or option_type == OptionTypes.EUROPEAN_PUT:

            if self._implementationType is BlackScholesTypes.DEFAULT:
                self._implementationType = BlackScholesTypes.ANALYTICAL

            if self._implementationType == BlackScholesTypes.ANALYTICAL:

                v = bs_value(spotPrice,
                             strike_price,
                             risk_free_rate,
                             time_to_expiry,
                             dividendRate,
                             volatility,
                             option_type.value)

                return v

            elif self._implementationType == BlackScholesTypes.BINOM_TREE:

                option_tree = BinomialTreeOption(spotPrice, strike_price, risk_free_rate,
                                                 time_to_expiry, self._num_step, dividendRate, volatility,
                                                 self._pu, self._pd, option_type)

                return option_tree.price()

            elif self._implementationType == BlackScholesTypes.CRR_TREE:

                option_tree = CRRTreeOption(spotPrice, strike_price, risk_free_rate,
                                            time_to_expiry, self._num_step, dividendRate, volatility,
                                            self._pu, self._pd, option_type)

                return option_tree.price()

            else:

                raise Exception("Implementation not available for this product")

        elif option_type == OptionTypes.AMERICAN_CALL or option_type == OptionTypes.AMERICAN_PUT:

            if self._implementationType is BlackScholesTypes.DEFAULT:
                self._implementationType = BlackScholesTypes.CRR_TREE

            if self._implementationType is BlackScholesTypes.CRR_TREE:

                option_tree = CRRTreeOption(spotPrice, strike_price, risk_free_rate,
                                            time_to_expiry, self._num_step, dividendRate, volatility,
                                            self._pu, self._pd, option_type)

                return option_tree.price()

            elif self._implementationType is BlackScholesTypes.BINOM_TREE:
                option_tree = BinomialTreeOption(spotPrice, strike_price, risk_free_rate,
                                                 time_to_expiry, self._num_step, dividendRate, volatility,
                                                 self._pu, self._pd, option_type)

                return option_tree.price()

        else:

            raise Exception("Wrong option type")


def bs_value(S, K, r, T, q, sigma, option_type):
    """This function calculates the value of the European option based on Black-Scholes-Merton formula
    :param S: Asset price
    :param K: Strike price
    :param T: Time to Maturity
    :param r: risk-free rate (treasury bills)
    :param q: dividend yield
    :param sigma: volatility
    :param option_type: call or put option
    :return : option price
    """
    # determine N(d1) and N(d2)
    d1 = 1 / (sigma * np.sqrt(T)) * (np.log(S / K) + (r - q + sigma ** 2 / 2) * T)
    d2 = d1 - sigma * np.sqrt(T)
    # return based on optionType param
    if option_type == OptionTypes.EUROPEAN_CALL.value:
        phi = 1.0
    elif option_type == OptionTypes.EUROPEAN_PUT.value:
        phi = -1.0
    else:
        print('Wrong option type specified')
        return 0

    val = phi * norm.cdf(phi * d1) * S * np.exp(-q * T) - phi * norm.cdf(phi * d2) * K * np.exp(-r * T)
    return val


def bs_vega(S, K, r, T, q, sigma):
    """"
    :param S: Asset price
    :param K: Strike price
    :param T: Time to Maturity
    :param r: risk-free rate (treasury bills)
    :param q: dividend yield
    :param sigma: volatility
    :return: partial derivative w.r.t volatility
    """
    d1 = 1 / (sigma * np.sqrt(T)) * (np.log(S / K) + (r - q + sigma ** 2 / 2) * T)
    return S * np.sqrt(-q * T) * norm.pdf(d1)


def bs_delta(S, K, r, T, q, sigma, option_type):
    """"
    :param S: Asset price
    :param K: Strike price
    :param T: Time to Maturity
    :param r: risk-free rate (treasury bills)
    :param q: dividend yield
    :param sigma: volatility
    :param option_type:
    :return: partial derivative w.r.t spot price
    """

    if option_type == OptionTypes.EUROPEAN_CALL:
        phi = +1.0
    elif option_type == OptionTypes.EUROPEAN_PUT:
        phi = -1.0
    else:
        raise Exception("Unknown option type value")

    return phi * norm.cdf(phi * np.log(S / K) + (r - q + sigma ** 2 / 2) * T)


def bs_gamma(S, K, r, T, q, sigma):
    """"
        :param S: Asset price
        :param K: Strike price
        :param T: Time to Maturity
        :param r: risk-free rate (treasury bills)
        :param q: dividend yield
        :param sigma: volatility
        :return: partial derivative w.r.t delta
        """
    d1 = 1 / (sigma * np.sqrt(T)) * (np.log(S / K) + (r - q + sigma ** 2 / 2) * T)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_theta(S, K, r, T, q, sigma, option_type):
    """"
    :param S: Asset price
    :param K: Strike price
    :param T: Time to Maturity
    :param r: risk-free rate (treasury bills)
    :param q: dividend yield
    :param sigma: volatility
    :param option_type:
    :return: partial derivative w.r.t time
    """
    if option_type == OptionTypes.EUROPEAN_CALL:
        phi = +1.0
    elif option_type == OptionTypes.EUROPEAN_PUT:
        phi = -1.0
    else:
        raise Exception("Unknown option type value")

    d1 = 1 / (sigma * np.sqrt(T)) * (np.log(S / K) + (r - q + sigma ** 2 / 2) * T)
    d2 = d1 - sigma * np.sqrt(T)

    theta = - S * np.exp(-q * T) * norm.cdf(d1) * sigma / 2.0
    theta = theta - phi * r * K * np.exp(-r * T) * norm.cdf(phi * d2)
    theta = theta + phi * q * S * np.exp(-q * T) * norm.cdf(phi * d1)
    return theta


def bs_rho(S, K, r, T, q, sigma, option_type):
    """"
    :param S: Asset price
    :param K: Strike price
    :param T: Time to Maturity
    :param r: risk-free rate (treasury bills)
    :param q: dividend yield
    :param sigma: volatility
    :param option_type:
    :return: partial derivative w.r.t risk-free rate
    """
    if option_type == OptionTypes.EUROPEAN_CALL:
        phi = 1.0
    elif option_type == OptionTypes.EUROPEAN_PUT:
        phi = -1.0
    else:
        raise Exception("Unknown option type value")

    d1 = 1 / (sigma * np.sqrt(T)) * (np.log(S / K) + (r - q + sigma ** 2 / 2) * T)
    d2 = d1 - sigma * np.sqrt(T)

    rho = phi * K * T * np.exp(-r * T) * norm.cdf(phi * d2)
    return rho
