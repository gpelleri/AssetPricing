import unittest
from assetpricing.products.equity import Stock, BarrierOption
from assetpricing.products.underlying import Underlying
from assetpricing.utils.global_types import *

notional = 1.0
num_observations_per_year = 100


class EquityBarrierTest(unittest.TestCase):
    def setUp(self):
        self.r = 0.05
        self.q = 0.02

    def test_up_and_out(self):
        expiry = 1
        stock_price = 80.0
        vol = 0.20
        B = 110.0
        K = 100.0
        edf = Stock("EDF", True, stock_price, vol, self.q)
        ul = Underlying(edf)

        edf_call = BarrierOption("EDF-call", expiry, ul, K, B, EquityBarrierTypes.UP_AND_OUT_CALL)
        self.assertEqual(edf_call.value(self.r, num_observations_per_year, ul.get_price(), notional).round(4), 0.1789)

        edf_put = BarrierOption("EDF-put", expiry, ul, K, B, EquityBarrierTypes.UP_AND_OUT_PUT)
        self.assertEqual(edf_put.value(self.r, num_observations_per_year, ul.get_price(), notional).round(4), 18.1445)

    def test_up_and_in(self):
        expiry = 1
        stock_price = 80.0
        vol = 0.20
        B = 120.0
        K = 105.0
        edf = Stock("EDF", True, stock_price, vol, self.q)
        ul = Underlying(edf)

        edf_call = BarrierOption("EDF-call", expiry, ul, K, B, EquityBarrierTypes.UP_AND_IN_CALL)
        self.assertEqual(edf_call.value(self.r, num_observations_per_year, ul.get_price(), notional).round(4), 0.6898)

        edf_put = BarrierOption("EDF-put", expiry, ul, K, B, EquityBarrierTypes.UP_AND_IN_PUT)
        self.assertEqual(edf_put.value(self.r, num_observations_per_year, ul.get_price(), notional).round(4), 0.0144)

    def test_down_and_in(self):
        expiry = 1
        stock_price = 80.0
        vol = 0.20
        B = 120
        K = 90
        edf = Stock("EDF", True, stock_price, vol, self.q)
        ul = Underlying(edf)

        edf_call = BarrierOption("EDF-call", expiry, ul, K, B, EquityBarrierTypes.DOWN_AND_IN_CALL)
        self.assertEqual(edf_call.value(self.r, num_observations_per_year, ul.get_price(), notional).round(4), 3.5522)

        edf_put = BarrierOption("EDF-put", expiry, ul, K, B, EquityBarrierTypes.DOWN_AND_IN_PUT)
        self.assertEqual(edf_put.value(self.r, num_observations_per_year, ul.get_price(), notional).round(4), 10.747)

    def test_down_and_out(self):
        expiry = 1
        stock_price = 80.0
        vol = 0.20
        B = 70
        K = 85
        edf = Stock("EDF", True, stock_price, vol, self.q)
        ul = Underlying(edf)

        edf_call = BarrierOption("EDF-call", expiry, ul, K, B, EquityBarrierTypes.DOWN_AND_OUT_CALL)
        self.assertEqual(edf_call.value(self.r, num_observations_per_year, ul.get_price(), notional).round(4), 4.9045)

        edf_put = BarrierOption("EDF-put", expiry, ul, K, B, EquityBarrierTypes.DOWN_AND_OUT_PUT)
        self.assertEqual(edf_put.value(self.r, num_observations_per_year, ul.get_price(), notional).round(4), 1.0780)

