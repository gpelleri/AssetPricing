import math
import numpy as np

""" 
Price an option by the binomial CRR model 
"""
# TODO : WIP - better to make it without using a class ? straight functions might be better


class BinomialTreeOption(object):
    def __init__(
            self, S0, K, r=0.05, T=1, N=2, pu=0.0, pd=0.0,
            div=0, sigma=0, is_put=False, is_am=False):
        """
        Initialize the tree base class.
        Defaults to European call unless specified.

        :param S0: initial stock price
        :param K: strike price
        :param r: risk-free interest rate
        :param T: time to maturity
        :param N: number of time steps
        :param pu: probability at up state
        :param pd: probability at down state
        :param div: Dividend yield
        :param is_put: True for a put option,
                False for a call option
        :param is_am: True for an American option,
                False for a European option
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.N = max(1, N)
        self.STs = []  # Declare the stock prices tree

        """ Optional parameters used by derived classes """
        self.pu = pu
        self.pd = pd
        self.div = div
        self.sigma = sigma
        self.is_call = not is_put
        self.is_european = not is_am

        self.setup_parameters()

    @property
    def dt(self):
        """ Single time step, in years """
        return self.T / float(self.N)

    @property
    def df(self):
        """ The discount factor """
        return math.exp(-(self.r - self.div) * self.dt)

    def setup_parameters(self):
        self.u = 1 + self.pu  # Expected value in the up state
        self.d = 1 - self.pd  # Expected value in the down state
        self.qu = (math.exp(
            (self.r - self.div) * self.dt) - self.d) / (self.u - self.d)
        self.qd = 1 - self.qu

    def init_stock_price_tree(self):
        # Initialize a 2D tree at T=0
        self.STs = [np.array([self.S0])]

        # Simulate the possible stock prices path
        for i in range(self.N):
            prev_branches = self.STs[-1]
            st = np.concatenate(
                (prev_branches * self.u,
                 [prev_branches[-1] * self.d]))
            self.STs.append(st)  # Add nodes at each time step

    def init_payoffs_tree(self):
        if self.is_call:
            return np.maximum(0, self.STs[self.N] - self.K)
        else:
            return np.maximum(0, self.K - self.STs[self.N])

    def check_early_exercise(self, payoffs, node):
        if self.is_call:
            return np.maximum(payoffs, self.STs[node] - self.K)
        else:
            return np.maximum(payoffs, self.K - self.STs[node])

    def traverse_tree(self, payoffs):
        for i in reversed(range(self.N)):
            # The payoffs from NOT exercising the option
            payoffs = (payoffs[:-1] * self.qu +
                       payoffs[1:] * self.qd) * self.df

            # Payoffs from exercising, for American options
            if not self.is_european:
                payoffs = self.check_early_exercise(payoffs, i)

        return payoffs

    def begin_tree_traversal(self):
        payoffs = self.init_payoffs_tree()
        return self.traverse_tree(payoffs)

    def price(self):
        """  The pricing implementation """
        self.setup_parameters()
        self.init_stock_price_tree()
        payoffs = self.begin_tree_traversal()
        return payoffs[0]


# if __name__ == '__main__':
#     am_option = BinomialTreeOption(50, 52,
#                                    r=0.05, T=2, N=2, pu=0.2, pd=0.2, is_put=True, is_am=True)
#
#     print("American put option price is:", am_option.price())

