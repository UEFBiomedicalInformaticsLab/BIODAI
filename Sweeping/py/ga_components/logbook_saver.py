from abc import ABC, abstractmethod
from deap.tools import Logbook
from pandas import DataFrame


def logbook_to_df(logbook: Logbook):
    header = logbook.header  # First level labels
    chapter_keys = logbook.chapters.keys()  # First level labels that have a second level
    non_chapter_keys = [k for k in header if k not in chapter_keys]  # First level labels without a second level

    columns = {}
    for nck in non_chapter_keys:
        columns[nck] = logbook.select(nck)

    for ck in chapter_keys:
        sub_chapter_keys = logbook.chapters[ck].header
        for sck in sub_chapter_keys:
            col_name = ck + "_" + sck
            columns[col_name] = logbook.chapters[ck].select(sck)

    df = DataFrame.from_dict(columns)
    return df


def logbook_to_csv(logbook: Logbook, file: str):
    df_log = logbook_to_df(logbook)
    df_log.to_csv(file, index=False)


class LogbookSaver(ABC):

    @abstractmethod
    def save(self, logbook: Logbook):
        raise NotImplementedError()


class DummyLogbookSaver(LogbookSaver):

    def save(self, logbook: Logbook):
        pass


class CsvLogbookSaver(LogbookSaver):
    __file: str

    def __init__(self, file: str):
        self.__file = file

    def save(self, logbook: Logbook):
        logbook_to_csv(logbook=logbook, file=self.__file)
