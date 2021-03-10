from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from BApp.ais.deeplabV3 import DeeplabV3
from BApp.uis import BAppui
import numpy as np
import cv2 as cv

class BMainWindow(QMainWindow):

    def __init__(self):
        super(BMainWindow, self).__init__()
        self.model = DeeplabV3() # 初始化模型

        self.bwd = BAppui.Ui_MainWindow()
        self.bwd.setupUi(self)

        self.bwd.originimg.triggered.connect(self.open_originimg)
        self.bwd.newbackground.triggered.connect(self.open_newbackground)
        self.bwd.clearbackground.triggered.connect(self.clear_background)
        self.bwd.start.clicked.connect(self.change_background)
        self.bwd.save.clicked.connect(self.save_img)

        self.newbackgroundflag = False
        self.newbackground = QPixmap()
    
    def open_originimg(self):
        file_name = QFileDialog.getOpenFileName(self, "选择原始图像", filter="图片文件(*.jpg *.png)")
        file_name = file_name[0]
        if file_name == "":
            return
        else:
            img = QPixmap(file_name)
            img = img.scaledToWidth(640)
            self.bwd.origin_jpg.setPixmap(img)

    def open_newbackground(self):
        file_name = QFileDialog.getOpenFileName(self, "选择背景", filter="图片文件(*.jpg *.png)")
        file_name = file_name[0]
        if file_name == "":
            return
        else:
            img = QPixmap(file_name)
            img = img.scaledToWidth(640)
            self.bwd.target_jpg.setPixmap(img)
            self.newbackground = img
            self.newbackgroundflag = True
    
    def clear_background(self):
        self.newbackgroundflag = False
        self.bwd.target_jpg.setText("请选择背景图片（默认为纯黑背景）")

    def change_background(self):
        origin_img = self.bwd.origin_jpg.pixmap()
        if origin_img is None:
            QMessageBox.information(self, "请选择原始图像", "请选择原始图像！", QMessageBox.Ok)
        else:
            origin_img = origin_img.toImage()
            origin_img = self.QImage2numpy(origin_img)
            res = self.model.image_segmentation(origin_img)
            res = res.cpu()
            target_img = np.zeros_like(origin_img)
            if self.newbackgroundflag == True:
                target_img = self.newbackground.scaledToWidth(640)
                target_img = target_img.toImage()
                target_img = self.QImage2numpy(target_img)
            target_img[res == 15] = origin_img[res == 15]
            target_img = self.numpy2QImage(target_img)
            target_img = QPixmap(target_img)
            self.bwd.target_jpg.setPixmap(target_img)
        
    def save_img(self):
        file_name = QFileDialog.getSaveFileName(self, "保存文件", "result.jpg", "图片文件(*.jpg *.png)")
        file_name = file_name[0]
        if file_name == "":
            return
        else:
            img = self.bwd.target_jpg.pixmap()
            img.save(file_name, quality=100)
    
    def numpy2QImage(self, img):
        height, width, bytesPerComponent = img.shape
        bytesPerLine = 3 * width
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)

    def QImage2numpy(self, img):
        size = img.size()
        s = img.bits().asstring(size.width() * size.height() * img.depth() // 8)
        arr = np.fromstring(s, dtype=np.uint8).reshape((size.height(), size.width(), img.depth() // 8))
        arr = arr[:,:,0:3]
        return arr
