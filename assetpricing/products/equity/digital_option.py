import numpy as np

from assetpricing.models.montecarlo import Montecarlo
from assetpricing.models.black_scholes import BlackScholes
from assetpricing.products.equity import Stock
from assetpricing.products.equity.option import Option
from assetpricing.utils.global_types import *
from scipy.stats import norm


class DigitalOption(Option):
    """
    Class that defines Digital Option and their pricing method.
    Please note it only prices European Digital Option (observation is made at Expiry)
    """
    def __init__(self, name, expiry, underlying, barrier, option_type: DigitalOptionTypes):
        super().__init__(name, expiry, underlying, strike=0)
        self.option_type = option_type
        self._barrier = barrier

    def get_option_type(self):
        return self.option_type

    def get_barrier(self):
        return self._barrier

    # Overload getStrike as digit have no strike but only a barrier
    def get_strike(self):
        return self._barrier

    def value(self, risk_free_rate, notional=1):
        ul = self.get_underlying()

        if np.any(ul.get_price() <= 0.0):
            raise Exception("Stock price must be greater than zero.")

        if np.any(self.get_expiry() < 0.0):
            raise Exception("Time to expiry must be positive.")

        S = ul.get_price()
        sigma = ul.get_vol()
        # no strike on digital
        H = self._barrier

        r = risk_free_rate
        q = ul.getDiv()

        d1 = (np.log(S / H) + (r - q + sigma * sigma / 2.0) * self.getExpiry())
        d1 = d1 / sigma / np.sqrt(self.getExpiry())
        d2 = d1 - sigma * np.sqrt(self.getExpiry())

        if self.get_option_type() == DigitalOptionTypes.CASH_OR_NOTHING_CALL:
            v = np.exp(-r * self.getExpiry()) * norm.cdf(d2)
        elif self.get_option_type() == DigitalOptionTypes.CASH_OR_NOTHING_PUT:
            v = np.exp(-r * self.getExpiry()) * norm.cdf(-d2)
        elif self.get_option_type() == DigitalOptionTypes.ASSET_OR_NOTHING_CALL:
            v = S * np.exp(-q * self.getExpiry()) * norm.cdf(d1)
        elif self.get_option_type() == DigitalOptionTypes.ASSET_OR_NOTHING_PUT:
            v = S * np.exp(-q * self.getExpiry()) * norm.cdf(-d1)
        else:
            raise Exception("Unknown Digital option type.")

        return v * notional


# TODO : Create unit test module & include those & expand cases
if __name__ == '__main__':
    r = 0.05  # risk-free risk in annual %
    q = 0.01  # annual dividend rate

    edf = Stock("EDF", True, 100, 0.3, q)
    edf_call_digit = DigitalOption("EDF-call", 1, edf, 100, DigitalOptionTypes.CASH_OR_NOTHING_CALL)
    print(edf_call_digit.value(r))
    # should be equal to 0.469290

    edf_put_digit = DigitalOption("EDF-call", 1, edf, 100, DigitalOptionTypes.CASH_OR_NOTHING_PUT)
    print(edf_call_digit.value(r))
    # should be equal to 0.481939
