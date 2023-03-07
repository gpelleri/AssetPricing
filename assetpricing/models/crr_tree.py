import math

from assetpricing.models.binom_tree import BinomialTreeOption

""" 
Price an option by the binomial CRR model using previous binomial tree model
"""


class BinomialCRROption(BinomialTreeOption):
    def setup_parameters(self):
        self.u = math.exp(self.sigma * math.sqrt(self.dt))
        self.d = 1./self.u
        self.qu = (math.exp((self.r-self.div)*self.dt) -
                   self.d)/(self.u-self.d)
        self.qd = 1-self.qu


# if __name__ == '__main__':
#     eu_option = BinomialCRROption(
#     50, 52, r=0.05, T=2, N=2, sigma=0.3, is_put=True)
# #
#     print('European put:', eu_option.price())
# #
# # European put should be equal to : 6.245708445206436
# #
#     am_option = BinomialCRROption(50, 52, r=0.05, T=2, N=2, sigma=0.3, is_put=True, is_am=True)
# #
#     print('American put option price is:', am_option.price())
# #
# # American put option price should be : 7.428401902704834




