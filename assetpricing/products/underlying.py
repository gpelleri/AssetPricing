from assetpricing.products.security import Security
from assetpricing.products.equity.stock import Stock

#   TODO : WIP ; need to allow for other underlying type and therefore check type & cast when using type based method


class Underlying:
    def __init__(self, option: Stock):

        self._type = option.get_type()
        self._underlying = option

    def get_price(self):
        return self._underlying.get_price()

    def get_vol(self):
        return self._underlying.get_vol()

    def get_div(self):
        return self._underlying.get_div()
