from PyQt5.QtWidgets import QApplication
from BApp.uis.BMainWindow import BMainWindow
import sys

class BApplication(QApplication):

    def __init__(self):
        super().__init__(sys.argv)
        self.bwd = BMainWindow()
        self.bwd.show()