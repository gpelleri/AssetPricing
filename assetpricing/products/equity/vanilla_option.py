from assetpricing.models.black_scholes import *
from assetpricing.models.montecarlo import *
from assetpricing.products.equity.option import Option
from assetpricing.products.equity.stock import Stock
from assetpricing.utils.global_types import *


class VanillaOption(Option):
    """
    Class that defines the most simple European Option.
    """

    def __init__(self, name, expiry, underlying, strike, option_type: OptionTypes):
        super().__init__(name, expiry, underlying, strike)
        self.option_type = option_type

    def value(self,
              risk_free_rate: float,
              model):

        ul = self.getUnderlying()

        if np.any(ul.getPrice() <= 0.0):
            raise Exception("Stock price must be greater than zero.")

        if np.any(self.getExpiry() < 0.0):
            raise Exception("Time to expiry must be positive.")

        if isinstance(model, BlackScholes):

            value = model.value(ul.getPrice(), self.getStrike(), risk_free_rate, self.getExpiry(), ul.getDiv(),
                                ul.getVol(), self.getOptionType())

        elif isinstance(model, Montecarlo):
            value = model.bs_value_mc(ul.getPrice(), self.getStrike(), risk_free_rate, self.getExpiry(), ul.getDiv(),
                                      ul.getVol(), self.getOptionType())

        else:
            raise Exception("Model : " + model + " isn't implemented")

        return value


if __name__ == '__main__':
    r = 0.05  # risk-free risk in annual %
    q = 0  # annual dividend rate

    apple = Stock("AAPL", False)
    chains = apple.getOptionData()
    chains['Dividend'] = apple.getDiv()
    chains['Spot'] = apple.getPrice()
    chains['Risk-Free Rate'] = r

    cleaned = apple.clean_Option_Data(chains)

    x,y,z = apple.build_Impl_Vol_Surface(cleaned, OptionTypes.EUROPEAN_PUT)
    apple.plot_Vol_Surface(x,y,z)



