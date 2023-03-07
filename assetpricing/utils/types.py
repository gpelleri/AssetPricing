from enum import Enum

# TODO Create a class for derivative types


class SecurityType(Enum):
    STOCK = 0
    BOND = 1
    DERIVATIVE = 2
    INDEX = 3

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class OptionTypes(Enum):
    EUROPEAN_CALL = 1
    EUROPEAN_PUT = 2
    AMERICAN_CALL = 3
    AMERICAN_PUT = 4
    DIGITAL_CALL = 5
    DIGITAL_PUT = 6
    BARRIER = 7

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class EquityBarrierTypes(Enum):
    DOWN_AND_OUT_CALL = 1
    DOWN_AND_IN_CALL = 2
    UP_AND_OUT_CALL = 3
    UP_AND_IN_CALL = 4
    UP_AND_OUT_PUT = 5
    UP_AND_IN_PUT = 6
    DOWN_AND_OUT_PUT = 7
    DOWN_AND_IN_PUT = 8


class BlackScholesTypes(Enum):
    DEFAULT = 0
    ANALYTICAL = 1
    BINOM_TREE = 2
    CRR_TREE = 3


# useless if i don't implement other deri than option
class DerivativeType(Enum):
    OPTION = 0
