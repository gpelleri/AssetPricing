from assetpricing.models.montecarlo import *
from assetpricing.products.equity import Stock
from assetpricing.products.equity.option import Option
from assetpricing.utils.global_types import *
import matplotlib.pyplot as plt


class BarrierOption(Option):
    """
    Class that defines Barrier Option and their pricing method.
    """

    def __init__(self, name, expiry, underlying, strike, barrier, barrier_type: EquityBarrierTypes):
        super().__init__(name, expiry, underlying, strike)
        self.option_type = OptionTypes.BARRIER
        self.barrier_type = barrier_type
        self._barrier = barrier

    def getBarrierType(self):
        return self.barrier_type

    def getBarrier(self):
        return self._barrier

    def value(self,
              risk_free_rate: float,
              num_observations,
              prices=None,
              notional=1):
        """ Values a barrier option according to formulas found in Bouzoubaa.
         Barrier adjustment for discrete observation values are told to be taken from Broadie - 1977

         :param risk_free_rate: risk-free rate
         :param num_observations : nb of observations per year
         :param prices: this allows to pass an array of prices in case when want to compute multiple prices at once.
         It is especially necessary when plotting payoffs
         :param notional: notion of the option
        """

        ul = self.getUnderlying()
        if prices is None:
            stock_price = ul.getPrice()
        else:
            stock_price = prices

        if np.any(stock_price <= 0.0):
            raise Exception("Stock price must be greater than zero.")

        if self.getExpiry() < 0.0:
            raise Exception("Time to expiry must be greater than 0")

        if isinstance(stock_price, int):
            stock_price = float(stock_price)

        if isinstance(stock_price, float):
            stock_prices = [stock_price]
        else:
            stock_prices = stock_price

        values = []
        for s in stock_prices:
            v = value_bs(s, self.strike, self.getExpiry(), ul.getDiv(), ul.getVol(), risk_free_rate, self.getBarrier(),
                         self.getBarrierType(), num_observations, notional)
            values.append(v)

        if isinstance(stock_price, float):
            return values[0]
        else:
            return np.array(values)


def value_bs(stock_price: (float, np.ndarray),
             strike,
             expiry,
             div_rate,
             sigma,
             risk_free_rate: float,
             barrier_level,
             option_type: int,
             num_observations,  # number of observations per year
             notional=1):
    """ Values a barrier option according to formulas found in Bouzoubaa.
         Barrier adjustment for discrete observation values are told to be taken from Broadie - 1977 """

    lnS0k = np.log(stock_price / strike)
    sqrtT = np.sqrt(expiry)

    r = risk_free_rate
    q = div_rate

    K = strike
    S = stock_price
    H = barrier_level

    volatility = sigma
    sigma_root_t = volatility * sqrtT
    v2 = volatility * volatility
    mu = r - q
    d1 = (lnS0k + (mu + v2 / 2.0) * expiry) / sigma_root_t
    d2 = (lnS0k + (mu - v2 / 2.0) * expiry) / sigma_root_t
    df = np.exp(-r * expiry)
    dq = np.exp(-q * expiry)

    # call & put formula under BS to return faster if Barrier Option turns out to be a simple C or P
    c = S * dq * norm.cdf(d1) - K * df * norm.cdf(d2)
    p = K * df * norm.cdf(-d2) - S * dq * norm.cdf(-d1)

    if option_type == EquityBarrierTypes.DOWN_AND_OUT_CALL and S <= H:
        return 0.0
    elif option_type == EquityBarrierTypes.UP_AND_OUT_CALL and S >= H:
        return 0.0
    elif option_type == EquityBarrierTypes.UP_AND_OUT_PUT and S >= H:
        return 0.0
    elif option_type == EquityBarrierTypes.DOWN_AND_OUT_PUT and S <= H:
        return 0.0
    elif option_type == EquityBarrierTypes.DOWN_AND_IN_CALL and S <= H:
        return c
    elif option_type == EquityBarrierTypes.UP_AND_IN_CALL and S >= H:
        return c
    elif option_type == EquityBarrierTypes.UP_AND_IN_PUT and S >= H:
        return p
    elif option_type == EquityBarrierTypes.DOWN_AND_IN_PUT and S <= H:
        return p

    num_observations = 1 + expiry * num_observations

    # Adjusment to get from continuous to discrete observations. Taken from Bouzoubaa (which quotes Broadie)
    h_adj = H
    t = expiry / num_observations

    if option_type == EquityBarrierTypes.DOWN_AND_OUT_CALL:
        h_adj = H * np.exp(-0.5826 * sigma * np.sqrt(t))
    elif option_type == EquityBarrierTypes.DOWN_AND_IN_CALL:
        h_adj = H * np.exp(-0.5826 * sigma * np.sqrt(t))
    elif option_type == EquityBarrierTypes.UP_AND_IN_CALL:
        h_adj = H * np.exp(0.5826 * sigma * np.sqrt(t))
    elif option_type == EquityBarrierTypes.UP_AND_OUT_CALL:
        h_adj = H * np.exp(0.5826 * sigma * np.sqrt(t))
    elif option_type == EquityBarrierTypes.UP_AND_IN_PUT:
        h_adj = H * np.exp(0.5826 * sigma * np.sqrt(t))
    elif option_type == EquityBarrierTypes.UP_AND_OUT_PUT:
        h_adj = H * np.exp(0.5826 * sigma * np.sqrt(t))
    elif option_type == EquityBarrierTypes.DOWN_AND_OUT_PUT:
        h_adj = H * np.exp(-0.5826 * sigma * np.sqrt(t))
    elif option_type == EquityBarrierTypes.DOWN_AND_IN_PUT:
        h_adj = H * np.exp(-0.5826 * sigma * np.sqrt(t))
    else:
        raise Exception("Wrong Barrier Option type")

    # set back the adjusted barrier to the actual barrier
    H = h_adj
    # key computations
    lbd = (mu + v2 / 2.0) / v2
    y = np.log(H ** 2 / (S * K)) / sigma_root_t + lbd * sigma_root_t
    x1 = np.log(S / H) / sigma_root_t + lbd * sigma_root_t
    y1 = np.log(H / S) / sigma_root_t + lbd * sigma_root_t

    # Bouzoubaa tells us that :
    # C(K,T) = CUO + CUI
    # P(K,T) = PDO + PDI
    if option_type == EquityBarrierTypes.DOWN_AND_OUT_CALL:
        if H >= K:
            c_do = S * dq * norm.cdf(x1) - K * df * norm.cdf(x1 - sigma_root_t) \
                   - S * dq * pow((H / S), 2.0 * lbd) * norm.cdf(y1) \
                   + K * df * pow((H / S), 2.0 * lbd - 2.0) * norm.cdf(y1 - sigma_root_t)
            price = c_do
        else:
            c_di = S * dq * pow((H / S), 2.0 * lbd) * norm.cdf(y) \
                   - K * df * pow((H / S), 2.0 * lbd - 2.0) * norm.cdf(y - sigma_root_t)
            price = c - c_di
    elif option_type == EquityBarrierTypes.DOWN_AND_IN_CALL:
        if H <= K:
            c_di = S * dq * pow((H / S), 2.0 * lbd) * norm.cdf(y) \
                   - K * df * pow((H / S), 2.0 * lbd - 2.0) * norm.cdf(y - sigma_root_t)
            price = c_di
        else:
            c_do = S * dq * norm.cdf(x1) \
                   - K * df * norm.cdf(x1 - sigma_root_t) \
                   - S * dq * pow((H / S), 2.0 * lbd) * norm.cdf(y1) \
                   + K * df * pow((H / S), 2.0 * lbd - 2.0) * norm.cdf(y1 - sigma_root_t)
            price = c - c_do
    elif option_type == EquityBarrierTypes.UP_AND_IN_CALL:
        if H >= K:
            c_ui = S * dq * norm.cdf(x1) - K * df * norm.cdf(x1 - sigma_root_t) \
                   - S * dq * pow((H / S), 2.0 * lbd) * (norm.cdf(-y) - norm.cdf(-y1)) \
                   + K * df * pow((H / S), 2.0 * lbd - 2.0) * \
                   (norm.cdf(-y + sigma_root_t) - norm.cdf(-y1 + sigma_root_t))
            price = c_ui
        else:
            price = c
    elif option_type == EquityBarrierTypes.UP_AND_OUT_CALL:
        if H > K:
            c_ui = S * dq * norm.cdf(x1) - K * df * norm.cdf(x1 - sigma_root_t) \
                   - S * dq * pow((H / S), 2.0 * lbd) * (norm.cdf(-y) - norm.cdf(-y1)) \
                   + K * df * pow((H / S), 2.0 * lbd - 2.0) * \
                   (norm.cdf(-y + sigma_root_t) - norm.cdf(-y1 + sigma_root_t))
            price = c - c_ui
        else:
            price = 0.0
    elif option_type == EquityBarrierTypes.UP_AND_IN_PUT:
        if H > K:
            p_ui = -S * dq * pow((H / S), 2.0 * lbd) * norm.cdf(-y) \
                   + K * df * pow((H / S), 2.0 * lbd - 2.0) * norm.cdf(-y + sigma_root_t)
            price = p_ui
        else:
            p_uo = -S * dq * norm.cdf(-x1) \
                   + K * df * norm.cdf(-x1 + sigma_root_t) \
                   + S * dq * pow((H / S), 2.0 * lbd) * norm.cdf(-y1) \
                   - K * df * pow((H / S), 2.0 * lbd - 2.0) * \
                   norm.cdf(-y1 + sigma_root_t)
            price = p - p_uo
    elif option_type == EquityBarrierTypes.UP_AND_OUT_PUT:
        if H >= K:
            p_ui = -S * dq * pow((H / S), 2.0 * lbd) * norm.cdf(-y) \
                   + K * df * pow((H / S), 2.0 * lbd - 2.0) * norm.cdf(-y + sigma_root_t)
            price = p - p_ui
        else:
            p_uo = -S * dq * norm.cdf(-x1) \
                   + K * df * norm.cdf(-x1 + sigma_root_t) \
                   + S * dq * pow((H / S), 2.0 * lbd) * norm.cdf(-y1) \
                   - K * df * pow((H / S), 2.0 * lbd - 2.0) * \
                   norm.cdf(-y1 + sigma_root_t)
            price = p_uo
    elif option_type == EquityBarrierTypes.DOWN_AND_OUT_PUT:
        if H >= K:
            price = 0.0
        else:
            p_di = -S * dq * norm.cdf(-x1) \
                   + K * df * norm.cdf(-x1 + sigma_root_t) \
                   + S * dq * pow((H / S), 2.0 * lbd) * (norm.cdf(y) - norm.cdf(y1)) \
                   - K * df * pow((H / S), 2.0 * lbd - 2.0) * \
                   (norm.cdf(y - sigma_root_t) - norm.cdf(y1 - sigma_root_t))
            price = p - p_di
    elif option_type == EquityBarrierTypes.DOWN_AND_IN_PUT:
        if H >= K:
            price = p
        else:
            p_di = -S * dq * norm.cdf(-x1) \
                   + K * df * norm.cdf(-x1 + sigma_root_t) \
                   + S * dq * pow((H / S), 2.0 * lbd) * (norm.cdf(y) - norm.cdf(y1)) \
                   - K * df * pow((H / S), 2.0 * lbd - 2.0) * \
                   (norm.cdf(y - sigma_root_t) - norm.cdf(y1 - sigma_root_t))
            price = p_di
    else:
        raise Exception("Unknown Barrier Option type")

    return price * notional


if __name__ == '__main__':
    r = 0.05
    q = 0.01
    expiry = 0.5
    edf = Stock("EDF", True, 100, 0.25, q)
    edf_call = BarrierOption("EDF-Call", 1e-6, edf, 1.3, 1.45, EquityBarrierTypes.UP_AND_OUT_CALL)
    stock_prices = np.linspace(1.2, 1.6, 100)
    values = edf_call.value(r, 1, 1, stock_prices)

    plt.figure(figsize=(8, 5))
    plt.plot(stock_prices, values)
    plt.xlabel("Stock Prices")
    plt.ylabel("Value")
    plt.show()