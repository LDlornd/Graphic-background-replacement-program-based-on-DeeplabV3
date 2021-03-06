# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'BApp.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1600, 914)
        MainWindow.setMinimumSize(QtCore.QSize(1600, 800))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setMinimumSize(QtCore.QSize(1600, 900))
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.gridLayout.setContentsMargins(20, 20, 20, 20)
        self.gridLayout.setSpacing(20)
        self.gridLayout.setObjectName("gridLayout")
        self.origin_jpg = QtWidgets.QLabel(self.centralwidget)
        self.origin_jpg.setMinimumSize(QtCore.QSize(640, 480))
        self.origin_jpg.setMaximumSize(QtCore.QSize(640, 16777215))
        self.origin_jpg.setLineWidth(1)
        self.origin_jpg.setAlignment(QtCore.Qt.AlignCenter)
        self.origin_jpg.setObjectName("origin_jpg")
        self.gridLayout.addWidget(self.origin_jpg, 1, 0, 1, 1)
        self.save = QtWidgets.QPushButton(self.centralwidget)
        self.save.setObjectName("save")
        self.gridLayout.addWidget(self.save, 2, 2, 1, 1)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setObjectName("line")
        self.gridLayout.addWidget(self.line, 1, 1, 2, 1)
        self.start = QtWidgets.QPushButton(self.centralwidget)
        self.start.setObjectName("start")
        self.gridLayout.addWidget(self.start, 2, 0, 1, 1)
        self.target_jpg = QtWidgets.QLabel(self.centralwidget)
        self.target_jpg.setMinimumSize(QtCore.QSize(640, 480))
        self.target_jpg.setMaximumSize(QtCore.QSize(640, 16777215))
        self.target_jpg.setAlignment(QtCore.Qt.AlignCenter)
        self.target_jpg.setObjectName("target_jpg")
        self.gridLayout.addWidget(self.target_jpg, 1, 2, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1600, 22))
        self.menubar.setObjectName("menubar")
        self.file = QtWidgets.QMenu(self.menubar)
        self.file.setObjectName("file")
        self.Menu = QtWidgets.QMenu(self.file)
        self.Menu.setObjectName("Menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.originimg = QtWidgets.QAction(MainWindow)
        self.originimg.setObjectName("originimg")
        self.newbackground = QtWidgets.QAction(MainWindow)
        self.newbackground.setObjectName("newbackground")
        self.clearbackground = QtWidgets.QAction(MainWindow)
        self.clearbackground.setObjectName("clearbackground")
        self.Menu.addAction(self.newbackground)
        self.Menu.addAction(self.clearbackground)
        self.file.addAction(self.originimg)
        self.file.addAction(self.Menu.menuAction())
        self.menubar.addAction(self.file.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "??????????????????"))
        self.origin_jpg.setText(_translate("MainWindow", "?????????????????????"))
        self.save.setText(_translate("MainWindow", "????????????"))
        self.start.setText(_translate("MainWindow", "????????????"))
        self.target_jpg.setText(_translate("MainWindow", "????????????????????????????????????????????????"))
        self.file.setTitle(_translate("MainWindow", "????????????"))
        self.Menu.setTitle(_translate("MainWindow", "????????????"))
        self.originimg.setText(_translate("MainWindow", "????????????"))
        self.newbackground.setText(_translate("MainWindow", "??????????????????"))
        self.clearbackground.setText(_translate("MainWindow", "????????????"))
