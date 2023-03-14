from assetpricing.models.montecarlo import *
from assetpricing.products.equity.option import Option
from assetpricing.utils.types import *


class BarrierOption(Option):
    """
    Class that defines Barrier Option and their pricing method.
    """
    def __init__(self, name, expiry, underlying, strike, barrier, option_type: OptionTypes):
        super().__init__(name, expiry, underlying, strike)
        self.option_type = option_type
        self._barrier = barrier

    def getOptionType(self):
        return self.option_type

    def getBarrier(self):
        return self._barrier

    # Doesn't Belong to BS class as it would expand too much BlackScholes value entry function
    # Maybe to a proper model - not a big fan either
    def value(self,
              option_type: int,
              num_observations,  # number of observations per year
              risk_free_rate: float):
        """ Values a barrier option according to formulas found in Bouzoubaa.
         Barrier adjustment for discrete observation values are told to be taken from Broadie - 1977 """

        ul = self.getUnderlying()
        # we compute number of intermediate values that we're going to use a lot
        # I believe for performance & comprehension using getters 50 times isn't great
        r = risk_free_rate
        q = ul.getDiv()
        K = self.getStrike()
        S = ul.getPrice()
        H = self.getBarrier()
        volatility = ul.getVol()

        if np.any(S <= 0.0):
            raise Exception("Stock price must be greater than zero.")

        if self.getExpiry() < 0.0:
            raise Exception("Time to expiry must be greater than 0")

        sigma_root_t = volatility * np.sqrt(self.getExpiry())
        v2 = volatility ** 2
        mu = r - q
        d1 = (np.log(ul.getPrice() / self.getStrike()) + (mu + v2 / 2.0) * self.getExpiry()) / sigma_root_t
        d2 = (np.log(ul.getPrice() / self.getStrike()) + (mu - v2 / 2.0) * self.getExpiry()) / sigma_root_t
        df = np.exp(-r * self.getExpiry())
        dq = np.exp(-q * self.getExpiry())

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

        num_observations = 1 + self.getExpiry() * num_observations

        # Adjusment to get from continuous to discrete observations. Taken from Bouzoubaa (which quotes Broadie)
        h_adj = H
        t = self.getExpiry() / num_observations

        if option_type == EquityBarrierTypes.DOWN_AND_OUT_CALL:
            h_adj = H * np.exp(-0.5826 * volatility * np.sqrt(t))
        elif option_type == EquityBarrierTypes.DOWN_AND_IN_CALL:
            h_adj = H * np.exp(-0.5826 * volatility * np.sqrt(t))
        elif option_type == EquityBarrierTypes.UP_AND_IN_CALL:
            h_adj = H * np.exp(0.5826 * volatility * np.sqrt(t))
        elif option_type == EquityBarrierTypes.UP_AND_OUT_CALL:
            h_adj = H * np.exp(0.5826 * volatility * np.sqrt(t))
        elif option_type == EquityBarrierTypes.UP_AND_IN_PUT:
            h_adj = H * np.exp(0.5826 * volatility * np.sqrt(t))
        elif option_type == EquityBarrierTypes.UP_AND_OUT_PUT:
            h_adj = H * np.exp(0.5826 * volatility * np.sqrt(t))
        elif option_type == EquityBarrierTypes.DOWN_AND_OUT_PUT:
            h_adj = H * np.exp(-0.5826 * volatility * np.sqrt(t))
        elif option_type == EquityBarrierTypes.DOWN_AND_IN_PUT:
            h_adj = H * np.exp(-0.5826 * volatility * np.sqrt(t))
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
        if option_type == EquityBarrierTypes.DOWN_AND_OUT_CALL.value:
            if H >= K:
                c_do = S * dq * norm.cdf(x1) - K * df * norm.cdf(x1 - sigma_root_t) \
                       - S * dq * pow((H / S), 2.0 * lbd) * norm.cdf(y1) \
                       + K * df * pow((H / S), 2.0 * lbd - 2.0) * norm.cdf(y1 - sigma_root_t)
                price = c_do
            else:
                c_di = S * dq * pow((H / S), 2.0 * lbd) * norm.cdf(y) \
                       - K * df * pow((H / S), 2.0 * lbd - 2.0) * norm.cdf(y - sigma_root_t)
                price = c - c_di
        elif option_type == EquityBarrierTypes.DOWN_AND_IN_CALL.value:
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
        elif option_type == EquityBarrierTypes.UP_AND_IN_CALL.value:
            if H >= K:
                c_ui = S * dq * norm.cdf(x1) - K * df * norm.cdf(x1 - sigma_root_t) \
                       - S * dq * pow((H / S), 2.0 * lbd) * (norm.cdf(-y) - norm.cdf(-y1)) \
                       + K * df * pow((H / S), 2.0 * lbd - 2.0) * \
                       (norm.cdf(-y + sigma_root_t) - norm.cdf(-y1 + sigma_root_t))
                price = c_ui
            else:
                price = c
        elif option_type == EquityBarrierTypes.UP_AND_OUT_CALL.value:
            if H > K:
                c_ui = S * dq * norm.cdf(x1) - K * df * norm.cdf(x1 - sigma_root_t) \
                       - S * dq * pow((H / S), 2.0 * lbd) * (norm.cdf(-y) - norm.cdf(-y1)) \
                       + K * df * pow((H / S), 2.0 * lbd - 2.0) * \
                       (norm.cdf(-y + sigma_root_t) - norm.cdf(-y1 + sigma_root_t))
                price = c - c_ui
            else:
                price = 0.0
        elif option_type == EquityBarrierTypes.UP_AND_IN_PUT.value:
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
        elif option_type == EquityBarrierTypes.UP_AND_OUT_PUT.value:
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
        elif option_type == EquityBarrierTypes.DOWN_AND_OUT_PUT.value:
            if H >= K:
                price = 0.0
            else:
                p_di = -S * dq * norm.cdf(-x1) \
                       + K * df * norm.cdf(-x1 + sigma_root_t) \
                       + S * dq * pow((H / S), 2.0 * lbd) * (norm.cdf(y) - norm.cdf(y1)) \
                       - K * df * pow((H / S), 2.0 * lbd - 2.0) * \
                       (norm.cdf(y - sigma_root_t) - norm.cdf(y1 - sigma_root_t))
                price = p - p_di
        elif option_type == EquityBarrierTypes.DOWN_AND_IN_PUT.value:
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

        return price
