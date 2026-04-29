import sys

class HierDRException(Exception):
    def __init__(self, error, sys_module):
        _, _, tb = sys_module.exc_info()
        self.file = tb.tb_frame.f_code.co_filename if tb else "unknown"
        self.line = tb.tb_lineno if tb else 0
        self.msg  = str(error)
        super().__init__(repr(self))

    def __repr__(self):
        return f"Error in [{self.file}] at line [{self.line}]: {self.msg}"
