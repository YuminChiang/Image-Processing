import os
import sys
import cv2
from PyQt5 import QtWidgets, QtGui, QtCore

from ui import Ui_MainWindow
from image_processor import *

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # init
        self.image = None
        self.current_views_scenes = {}
        
        self.assets_path = os.path.join(os.path.dirname(__file__), "../", "assets")

        self.setup_responsive_layout()

        # bind btn events
        self.ui.pushButton.clicked.connect(self.load_image)
        self.ui.pushButton_2.clicked.connect(self.apply_smooth)
        self.ui.pushButton_3.clicked.connect(self.apply_sharp)
        self.ui.pushButton_4.clicked.connect(self.apply_gaussian)
        self.ui.pushButton_5.clicked.connect(self.apply_lowpass)

    def setup_responsive_layout(self):
        central_widget = self.ui.centralwidget

        main_layout = QtWidgets.QVBoxLayout(central_widget)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(self.ui.pushButton)
        btn_layout.addWidget(self.ui.pushButton_2)
        btn_layout.addWidget(self.ui.pushButton_3)
        btn_layout.addWidget(self.ui.pushButton_4)
        btn_layout.addWidget(self.ui.pushButton_5)

        main_layout.addLayout(btn_layout)

        grid_layout = QtWidgets.QGridLayout()

        def add_view_block(label, view, row, col):
            v_layout = QtWidgets.QVBoxLayout()
            v_layout.addWidget(label)
            v_layout.addWidget(view) 
            grid_layout.addLayout(v_layout, row, col)

        add_view_block(self.ui.label, self.ui.graphicsView, 0, 0) 
        add_view_block(self.ui.label_2, self.ui.graphicsView_2, 0, 1)
        add_view_block(self.ui.label_3, self.ui.graphicsView_3, 1, 0) 
        add_view_block(self.ui.label_4, self.ui.graphicsView_4, 1, 1) 

        main_layout.addLayout(grid_layout)

    def reset(self):
        self.image = None
        self.ui.label.setText("Origin Image")
        self.ui.label_2.setText("Image")
        self.ui.label_3.setText("Image")
        self.ui.label_4.setText("Image")

        views = [self.ui.graphicsView, 
                 self.ui.graphicsView_2, 
                 self.ui.graphicsView_3, 
                 self.ui.graphicsView_4
                ]
        
        for view in views:
            view.setScene(None)
        
        self.current_views_scenes = {} 

    def display_image(self, img, view):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_image)

        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(pixmap)
        view.setScene(scene)
        
        view.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        
        self.current_views_scenes[view] = scene

    def resizeEvent(self, event):
        super(MainWindow, self).resizeEvent(event)
        
        for view, scene in self.current_views_scenes.items():
            if scene:
                view.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def load_image(self):
        self.reset()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Image",
            self.assets_path,
            "Image Files (*.png *.jpg *.bmp)"
        )
        if path:
            self.image = cv2.imread(path)
            self.display_image(self.image, self.ui.graphicsView)
    
    def apply_smooth(self):
        if self.image is not None:
            avgfilter = average_filter(self.image)
            medfilter = median_filter(self.image) 
            ftfilter = fft_denoise(self.image)
            self.display_image(avgfilter, self.ui.graphicsView_2)
            self.display_image(medfilter, self.ui.graphicsView_3)
            self.display_image(ftfilter, self.ui.graphicsView_4)
            self.ui.label_2.setText("1(a) Average filter")
            self.ui.label_3.setText("1(a) Median filter")
            self.ui.label_4.setText("1(b) Fourier transform")
    
    def apply_sharp(self):
        if self.image is not None:
            self.ui.graphicsView_2.setScene(None)
            if self.ui.graphicsView_2 in self.current_views_scenes:
                del self.current_views_scenes[self.ui.graphicsView_2]
                
            self.ui.label_2.setText("No use")
            sobel = sobel_sharp(self.image)
            ftsharp = fft_sharp(self.image)
            self.display_image(sobel, self.ui.graphicsView_3)
            self.display_image(ftsharp, self.ui.graphicsView_4)
            self.ui.label_3.setText("2(a) Sobel mask")
            self.ui.label_4.setText("2(b) Fourier transform")
    
    def apply_gaussian(self):
        if self.image is not None:
            for v in [self.ui.graphicsView_3, self.ui.graphicsView_4]:
                v.setScene(None)
                if v in self.current_views_scenes:
                    del self.current_views_scenes[v]

            self.ui.label_3.setText("No use")
            self.ui.label_4.setText("No use")
            gaussfilter = gauss_blur(self.image)
            self.display_image(gaussfilter, self.ui.graphicsView_2)
            self.ui.label_2.setText("Result")

    def apply_lowpass(self):
        if self.image is not None:
            for v in [self.ui.graphicsView_3, self.ui.graphicsView_4]:
                v.setScene(None)
                if v in self.current_views_scenes:
                    del self.current_views_scenes[v]

            self.ui.label_3.setText("No use")
            self.ui.label_4.setText("No use")
            lowpass = gauss_fft(self.image)
            self.display_image(lowpass, self.ui.graphicsView_2)
            self.ui.label_2.setText("Result")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())