import sys

__all__ = [
    "register_pdb_excepthook"
]


def register_pdb_excepthook():
    def excepthook(type, value, traceback):
        if type is KeyboardInterrupt:
            # don't capture keyboard interrupts (Ctrl+C)
            sys.__excepthook__(type, value, traceback)
        else:
            ipython_hook(type, value, traceback)

    # enter IPython debugger on exception
    from IPython.core import ultratb

    ipython_hook = ultratb.FormattedTB(mode='Context', color_scheme='Linux', call_pdb=1)
    sys.excepthook = excepthook
