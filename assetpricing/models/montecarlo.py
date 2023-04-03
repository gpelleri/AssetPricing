import numpy as np
from scipy.stats import norm

from assetpricing.utils.global_types import OptionTypes, EquityBarrierTypes
from math import exp, sqrt, log

"""
This class implements various montecarlo pricing functions.
"""


class Montecarlo():
    def __init__(self, num_paths=1000, nb_obs_year=1, seed=1234):
        """
        :param num_paths: number of paths to generate
        :param seed: to set a fixed seed - default value provided
        """
        self._num_paths = num_paths
        self._seed = seed
        self._nb_obs_year = nb_obs_year

    def getNumPath(self):
        return self._num_paths

    def getSeed(self):
        return self._seed

    def getObsPerYear(self):
        return self._nb_obs_year

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
        sqrT = np.sqrt(expiry)

        d = np.random.standard_normal(num_paths)
        ss = S0 * exp((r - q - sigma ** 2 / 2.0) * expiry)

        # +d / -d to have 2 payoff generated for each normal distribution draw
        if option_type == OptionTypes.EUROPEAN_CALL:

            for i in range(0, num_paths):
                s_1 = ss * exp(+d[i] * sigma * sqrT)
                s_2 = ss * exp(-d[i] * sigma * sqrT)
                payoff += max(s_1 - strike, 0.0)
                payoff += max(s_2 - strike, 0.0)

        elif option_type == OptionTypes.EUROPEAN_PUT:

            for i in range(0, num_paths):
                s_1 = ss * exp(+d[i] * sigma * sqrT)
                s_2 = ss * exp(-d[i] * sigma * sqrT)
                payoff += max(strike - s_1, 0.0)
                payoff += max(strike - s_2, 0.0)

        else:
            raise Exception("Unknown option type.")

        # remember to divide by 2 since we used + draw & - draw
        v = payoff * np.exp(-r * expiry) / num_paths / 2
        return v

    def barrier_mc(self, S, strike, risk_free, expiry, q, sigma, H, option_type, notional=1):

        nb_paths = self.getNumPath()

        if option_type == EquityBarrierTypes.DOWN_AND_OUT_CALL and S <= H:
            return 0.0
        elif option_type == EquityBarrierTypes.UP_AND_OUT_CALL and S >= H:
            return 0.0
        elif option_type == EquityBarrierTypes.UP_AND_OUT_PUT and S >= H:
            return 0.0
        elif option_type == EquityBarrierTypes.DOWN_AND_OUT_PUT and S <= H:
            return 0.0

        vanilla_call, vanilla_put = False, False

        if option_type == EquityBarrierTypes.DOWN_AND_IN_CALL.value and S <= H:
            vanilla_call = True
        elif option_type == EquityBarrierTypes.UP_AND_IN_CALL.value and S >= H:
            vanilla_call = True
        elif option_type == EquityBarrierTypes.UP_AND_IN_PUT.value and S >= H:
            vanilla_put = True
        elif option_type == EquityBarrierTypes.DOWN_AND_IN_PUT.value and S <= H:
            vanilla_put = True

        if vanilla_call or vanilla_put:
            stock_paths = generatePaths(nb_paths, 1, expiry, risk_free-q, S, sigma, self.getSeed())

        if vanilla_call:
            c = (np.maximum(stock_paths[:, -1] - strike, 0.0)).mean()
            c = c * np.exp(-risk_free * expiry)
            return c

        if vanilla_put:
            p = (np.maximum(strike - stock_paths[:, -1], 0.0)).mean()
            p = p * np.exp(-risk_free * expiry)
            return p

        # define constants & get full set of paths
        stock_paths = generatePaths(nb_paths, self._nb_obs_year, expiry, risk_free - q, S, sigma,
                                    self.getSeed())

        if option_type == EquityBarrierTypes.DOWN_AND_IN_CALL or \
                option_type == EquityBarrierTypes.DOWN_AND_OUT_CALL or \
                option_type == EquityBarrierTypes.DOWN_AND_IN_PUT or \
                option_type == EquityBarrierTypes.DOWN_AND_OUT_PUT:

            down_crossed = [False] * nb_paths

            for p in range(0, nb_paths):
                down_crossed[p] = np.any(stock_paths[p] <= H)

        if option_type == EquityBarrierTypes.UP_AND_IN_CALL.value or \
                option_type == EquityBarrierTypes.UP_AND_OUT_CALL.value or \
                option_type == EquityBarrierTypes.UP_AND_IN_PUT.value or \
                option_type == EquityBarrierTypes.UP_AND_OUT_PUT.value:

            up_crossed = [False] * nb_paths
            for p in range(0, nb_paths):
                up_crossed[p] = np.any(stock_paths[p] >= H)

        ones = np.ones(nb_paths)

        if option_type == EquityBarrierTypes.DOWN_AND_OUT_CALL.value:
            payoff = np.maximum(stock_paths[:, -1] - strike, 0.0) * \
                     (ones - down_crossed)
        elif option_type == EquityBarrierTypes.DOWN_AND_IN_CALL.value:
            payoff = np.maximum(stock_paths[:, -1] - strike, 0.0) * down_crossed
        elif option_type == EquityBarrierTypes.UP_AND_IN_CALL.value:
            payoff = np.maximum(stock_paths[:, -1] - strike, 0.0) * up_crossed
        elif option_type == EquityBarrierTypes.UP_AND_OUT_CALL.value:
            payoff = np.maximum(stock_paths[:, -1] - strike, 0.0) * \
                     (ones - up_crossed)
        elif option_type == EquityBarrierTypes.UP_AND_IN_PUT.value:
            payoff = np.maximum(strike - stock_paths[:, -1], 0.0) * up_crossed
        elif option_type == EquityBarrierTypes.UP_AND_OUT_PUT.value:
            payoff = np.maximum(strike - stock_paths[:, -1], 0.0) * \
                     (ones - up_crossed)
        elif option_type == EquityBarrierTypes.DOWN_AND_OUT_PUT.value:
            payoff = np.maximum(strike - stock_paths[:, -1], 0.0) * \
                     (ones - down_crossed)
        elif option_type == EquityBarrierTypes.DOWN_AND_IN_PUT.value:
            payoff = np.maximum(strike - stock_paths[:, -1], 0.0) * down_crossed
        else:
            raise Exception("Wrong barrier option type")

        v = payoff.mean() * np.exp(- risk_free * expiry)

        return v * notional


def generatePaths(nb_paths, nb_obs_year, expiry, mu, S, sigma, seed):
    """
    This function generates N paths of Y observations for a given stock price assuming a Geometric Brownian Motion
    :param nb_paths:
    :param nb_obs_year:
    :param expiry:
    :param mu:
    :param S:
    :param sigma:
    :param seed:
    :return:
    """
    np.random.seed(seed)
    dt = 1.0 / nb_obs_year
    # get the total nb of steps (in case expiry is larger than a year)
    num_time_steps = int(expiry / dt)
    vsqrt_dt = sigma * sqrt(dt)
    m = exp((mu - sigma * sigma / 2.0) * dt)

    # create empty numpy array
    stock_paths = np.empty((nb_paths, num_time_steps + 1))
    # fill 1st col of each row with starting stock price
    stock_paths[:, 0] = S
    # iterate over array to fill it with generated stock price
    for it in range(1, num_time_steps + 1):
        d = np.random.standard_normal((nb_paths))
        for ip in range(0, nb_paths):
            w = np.exp(d[ip] * vsqrt_dt)
            stock_paths[ip, it] = stock_paths[ip, it - 1] * m * w

    return stock_paths
