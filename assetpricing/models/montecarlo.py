import numpy as np
from scipy.stats import norm

from assetpricing.models.binom_tree import BinomialTreeOption
from assetpricing.models.crr_tree import CRRTreeOption
from assetpricing.utils.types import OptionTypes, BlackScholesTypes
from math import exp

"""
This class implements various montecarlo pricing functions.
"""


class Montecarlo():
    def __init__(self, num_paths=1000, seed=1234):
        """
        :param num_paths: number of paths to generate
        :param seed: to set a fixed seed - default value provided
        """
        self._num_paths = num_paths
        self._seed = seed
        pass

    def getNumPath(self):
        return self._num_paths

    def getSeed(self):
        return self._seed

    def bs_value_mc(self, S0, strike, r, expiry, q, sigma, option_type):
        """
        Montecarlo simulation for EU option
        :param S0: spot price
        :param strike: option strike
        :param expiry: maturity
        :param r: risk-free rate
        :param q: dividend rate
        :param sigma: volatility
        :param option_type: Euro call or euro put
        :return: option price
        """

        np.random.seed(self.getSeed())
        num_paths = self.getNumPath()
        payoff = 0.0

        d = np.random.standard_normal(num_paths)
        ss = S0 * exp((r - q - sigma ** 2 / 2.0) * expiry)

        # +g / -g to have 2 payoff generated for each normal distribution draw
        if option_type == OptionTypes.EUROPEAN_CALL:

            for i in range(0, num_paths):
                s_1 = ss * exp(+d[i] * sigma * np.sqrt(expiry))
                s_2 = ss * exp(-d[i] * sigma * np.sqrt(expiry))
                payoff += max(s_1 - strike, 0.0)
                payoff += max(s_2 - strike, 0.0)

        elif option_type == OptionTypes.EUROPEAN_PUT:

            for i in range(0, num_paths):
                s_1 = ss * exp(+d[i] * sigma * np.sqrt(expiry))
                s_2 = ss * exp(-d[i] * sigma * np.sqrt(expiry))
                payoff += max(strike - s_1, 0.0)
                payoff += max(strike - s_2, 0.0)

        else:
            raise Exception("Unknown option type.")

        # remember to divide by 2 since we used + draw & - draw
        v = payoff * np.exp(-r * expiry) / num_paths / 2
        return v

    def barrier_mc(self):
        pass
