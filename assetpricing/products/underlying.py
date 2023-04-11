from assetpricing.products.security import Security
from assetpricing.products.equity.stock import Stock

#   TODO : WIP ; need to allow for other underlying type and therefore check type & cast when using type based method


class Underlying:
    def __init__(self, option: Stock):

        self._type = option.getType()
        self._underlying = option

    def getPrice(self):
        return self._underlying.getPrice()

    def getVol(self):
        return self._underlying.getVol()

    def getDiv(self):
        return self._underlying.getDiv()
