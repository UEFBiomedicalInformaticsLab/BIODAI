class Reference:

    def __init__(self, value=None):
        self.__value = value

    def get_value(self):
        return self.__value

    def set_value(self, value):
        self.__value = value
