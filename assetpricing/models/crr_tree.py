import math

from assetpricing.models.binom_tree import BinomialTreeOption

""" 
Price an option by the binomial CRR model using previous binomial tree model
"""


class CRRTreeOption(BinomialTreeOption):
    def setup_parameters(self):
        self.u = math.exp(self.sigma * math.sqrt(self.dt))
        self.d = 1./self.u
        self.qu = (math.exp((self.r-self.div)*self.dt) -
                   self.d)/(self.u-self.d)
        self.qd = 1-self.qu





