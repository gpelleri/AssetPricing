import numpy as np
from scipy.stats import norm

from assetpricing.utils.types import OptionTypes, BlackScholesTypes


# This class implements Black & Scholes & Merton model


class BlackScholes():
    def __init__(self, implementationType):
        self._implementationType = implementationType

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
            # TODO IMPLEMENTER + TEST
                v = 0

            elif self._implementationType == BlackScholesTypes.CRR_TREE:
            # TODO IMPLEMENTER + TEST
                v = 0
                # crr_tree_val_avg(spotPrice,
                #                      risk_free_rate,
                #                      dividendRate,
                #                      self._volatility,
                #                      self._num_steps_per_year,
                #                      time_to_expiry,
                #                      option_type.value,
                #                      strike_price)['value']

                return v

            else:

                raise Exception("Implementation not available for this product")

        elif option_type == OptionTypes.AMERICAN_CALL or option_type == OptionTypes.AMERICAN_PUT:

            if self._implementationType is BlackScholesTypes.DEFAULT:
                self._implementationType = BlackScholesTypes.CRR_TREE

        # TODO IMPLEMENTER
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

    bs_value = phi * norm.cdf(phi * d1) * S * np.exp(-q * T) - phi * norm.cdf(phi * d2) * K * np.exp(-r * T)
    return bs_value


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


