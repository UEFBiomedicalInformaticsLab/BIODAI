from util.printer.printer import LogAndOutPrinter

global _default_log_and_out_printer
_default_log_and_out_printer = None


def default_log_and_out_printer():
    global _default_log_and_out_printer
    if _default_log_and_out_printer is None:
        _default_log_and_out_printer = LogAndOutPrinter(log_file="log.txt")
    return _default_log_and_out_printer


def log_and_out_print(s: str):
    default_log_and_out_printer().print(s=s)
