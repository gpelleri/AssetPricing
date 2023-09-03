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

    def get_name(self):
        return self.name

    def get_type(self):
        return self.type

    def get_price(self):
        """
        Return SECURITY PRICE ! This doesn't return underlying price in case of a derivatives !
        :return: security price
        """
        return self.price

