from ..utils import SecurityType
import uuid


class Security:
    def __init__(self, name, secType):
        if not SecurityType.has_value(secType):
            raise Exception("ValueError", "The Security type doesn't exist")
        else:
            self.type = secType
            self.name = name
            self._id = uuid.uuid1()
            self.price = 0

    def getName(self):
        return self.name

    def getType(self):
        return self.type

    def getPrice(self):
        """
        Return SECURITY PRICE ! This doesn't return underlying price in case of a derivatives !
        :return: security price
        """
        return self.price

