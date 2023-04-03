import unittest
from assetpricing.products.equity import Stock, VanillaOption
from assetpricing.utils.global_types import *
from assetpricing.models.black_scholes import BlackScholes
from assetpricing.models.montecarlo import Montecarlo


class EquityVanillaTest(unittest.TestCase):
    def setUp(self):
        self.r = 0.05
        self.q = 0.02

    def test_getValueBS(self):
        expiry = 0.5
        edf = Stock("EDF", True, 100, 0.25, self.q)
        md = BlackScholes(BlackScholesTypes.ANALYTICAL)

        edf_call = VanillaOption("EDF-call", expiry, edf, 110, OptionTypes.EUROPEAN_CALL)
        self.assertEqual(edf_call.value(self.r, md), 3.8597599507749933)

        edf_put = VanillaOption("EDF-put", expiry, edf, 110, OptionTypes.EUROPEAN_PUT)
        self.assertEqual(edf_put.value(self.r, md), 12.138866898974783)

    def test_getValueMC(self):
        mc = Montecarlo(1000000, 67889)
        expiry = 0.5
        edf = Stock("EDF", True, 100, 0.25, self.q)
        edf_call = VanillaOption("EDF-call", expiry, edf, 110, OptionTypes.EUROPEAN_CALL)
        # TODO : upgrade MC, lacking precision
        self.assertAlmostEqual(edf_call.value(self.r, mc), 3.85, 1)
