from assetpricing.models.black_scholes import BlackScholes
from assetpricing.products.equity import Stock
from assetpricing.products.equity.option import Option
from assetpricing.utils import OptionTypes, BlackScholesTypes
import numpy as np


class AmericanOption(Option):
    """
    Class that defines American (or European) Option and values it based on binomial or CRR tree
    """
    def __init__(self, name, expiry, underlying, strike, option_type: OptionTypes):
        super().__init__(name, expiry, underlying, strike)
        self.option_type = option_type

    def get_option_type(self):
        return self.option_type

    def value(self,
              risk_free_rate: float,
              model):

        ul = self.get_underlying()

        if np.any(ul.get_price() <= 0.0):
            raise Exception("Stock price must be greater than zero.")

        if np.any(self.get_expiry() < 0.0):
            raise Exception("Time to expiry must be positive.")

        if isinstance(model, BlackScholes):

            value = model.value(ul.get_price(), self.get_strike(), risk_free_rate, self.get_expiry(), ul.get_div(),
                                ul.get_vol(), self.get_option_type())

        else:
            raise Exception("Model : " + model + " isn't implemented")

        return value


# TODO : Create unit test module & include those & expand cases
if __name__ == '__main__':
    r = 0.05  # risk-free risk in annual %
    q = 0.02  # annual dividend rate
    edf = Stock("EDF", True, 50, 0, 0)

    # Test American put with Binomial tree
    edf_am_put = AmericanOption("EDF-put", 2, edf, 52, OptionTypes.AMERICAN_PUT)
    binom = BlackScholes(BlackScholesTypes.BINOM_TREE, 2, 0.2, 0.2)
    print(edf_am_put.value(r, binom))
    # 5.089632474198373

    # Test European call with Binomial tree
    edf_eu_call = AmericanOption("EDF-call", 2, edf, 52, OptionTypes.EUROPEAN_CALL)
    print(edf_eu_call.value(r, binom))
    # 7.141108542733969

    # Test American call with CRR
    total = Stock("Total", True, 50, 0.3, 0)
    total_am_call = AmericanOption("Total-call", 2, total, 52, OptionTypes.AMERICAN_CALL)
    crr = BlackScholes(BlackScholesTypes.CRR_TREE, 2)
    print(total_am_call.value(r, crr))
    # 9.194162707336549

    total_eu_put = AmericanOption("Total-put", 2, total, 52, OptionTypes.EUROPEAN_PUT)
    print(total_eu_put.value(r, crr))
    # 6.245708445206436
