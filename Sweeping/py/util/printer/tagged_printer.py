from util.printer.printer import Printer


class TaggedPrinter(Printer):
    __tag: str
    __inner: Printer

    def __init__(self, tag: str, inner: Printer):
        self.__tag = str(tag)
        self.__inner = inner

    def __add_tag(self, s: str):
        str_s = str(s)
        if '\n' in str_s or '\r' in str_s:
            return self.__tag + ">\n" + str_s
        else:
            return self.__tag + "> " + str_s

    def print(self, s: str):
        self.__inner.print(self.__add_tag(s))

    def title_print(self, s: str):
        self.__inner.print("\n" + self.__add_tag(str(s).upper()))

    def name(self) -> str:
        return self.__inner.name() + " with tag " + self.__tag

    def __str__(self) -> str:
        return str(self.__inner) + " with tag " + self.__tag
