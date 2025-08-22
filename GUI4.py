import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QThread
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (QPushButton, QApplication, QComboBox, QLabel, 
                            QFileDialog, QStatusBar, QMessageBox, QMainWindow,
                            QGridLayout, QVBoxLayout, QHBoxLayout, QWidget)

import pyqtgraph as pg
import sys
from process import Process
from webcam import Webcam
from video import Video
from interface import waitKey
from scipy.signal import butter, filtfilt
import os

# 设置 DeepFace 权重目录，避免自动下载
os.environ['DEEPFACE_HOME'] = r"D:\1-main\ces\deepface\weights"
from deepface import DeepFace

# ------------------- DeepFace 线程 -------------------
class DeepFaceThread(QThread):
    result_ready = QtCore.pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.frame_queue = []
        self.running = True

    def run(self):
        while self.running:
            if self.frame_queue:
                frame = self.frame_queue.pop(0)
                try:
                    result = DeepFace.analyze(
                        frame, actions=['age','gender','emotion','race'], enforce_detection=False
                    )
                    if isinstance(result, list) and len(result) > 0:
                        result = result[0]
                    self.result_ready.emit(result)
                except Exception as e:
                    print("DeepFace 分析失败:", e)
            else:
                self.msleep(50)

    def add_frame(self, frame):
        self.frame_queue.append(frame)

    def stop(self):
        self.running = False
        self.wait()

# ------------------- 主 GUI -------------------
class GUI(QMainWindow, QThread):
    def __init__(self):
        super(GUI,self).__init__()
        font = QFont()
        font.setFamily("SimHei")
        font.setPointSize(16)
        QApplication.setFont(font)
        
        self.initUI()
        self.webcam = Webcam()
        self.video = Video()
        self.input = self.webcam
        self.dirname = ""
        print("Input: webcam")
        self.statusBar.showMessage("Input: webcam",5000)
        self.btnOpen.setEnabled(False)
        self.process = Process()
        self.status = False
        self.frame = np.zeros((10,10,3),np.uint8)
        self.bpm = 0
        self.terminate = False
        
        # 生理指标变量
        self.rr = 0
        self.hrv = 0
        self.spo2 = 0
        self.hr_history = []
        self.red_signal, self.blue_signal = [], []
        self.bpm_history, self.rr_history, self.spo2_history = [], [], []

        # DeepFace 分析结果缓存
        self.age = None
        self.gender = None
        self.emotion = None
        self.race = None
        self.deepface_counter = 0  # 控制更新频率

        # 启动 DeepFace 线程
        self.deepface_thread = DeepFaceThread()
        self.deepface_thread.result_ready.connect(self.update_deepface_result)
        self.deepface_thread.start()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QGridLayout(central_widget)
        main_layout.setSpacing(10)
        font = QFont()
        font.setPointSize(16)
        
        control_layout = QHBoxLayout()
        control_layout.setSpacing(10)
        self.cbbInput = QComboBox()
        self.cbbInput.addItem("Webcam")
        self.cbbInput.addItem("Video")
        self.cbbInput.setCurrentIndex(0)
        self.cbbInput.setFixedHeight(50)
        self.cbbInput.setFont(font)
        self.cbbInput.activated.connect(self.selectInput)
        control_layout.addWidget(self.cbbInput, 1)
        self.btnOpen = QPushButton("Open")
        self.btnOpen.setFixedHeight(50)
        self.btnOpen.setFont(font)
        self.btnOpen.clicked.connect(self.openFileDialog)
        control_layout.addWidget(self.btnOpen, 1)
        self.btnStart = QPushButton("Start")
        self.btnStart.setFixedHeight(50)
        self.btnStart.setFont(font)
        self.btnStart.clicked.connect(self.run)
        control_layout.addWidget(self.btnStart, 1)
        
        self.lblDisplay = QLabel()
        self.lblDisplay.setStyleSheet("background-color: #000000")
        self.lblDisplay.setScaledContents(True)
        self.lblROI = QLabel()
        self.lblROI.setStyleSheet("background-color: #000000")
        self.lblROI.setScaledContents(True)
        
        info_layout = QVBoxLayout()
        info_layout.setSpacing(15)
        self.lblHR = QLabel("Frequency: ")
        self.lblHR.setFont(font)
        self.lblHR2 = QLabel("Heart rate: ")
        self.lblHR2.setFont(font)
        self.lblRR = QLabel("Respiratory rate: ")
        self.lblRR.setFont(font)
        self.lblHRV = QLabel("HRV: ")
        self.lblHRV.setFont(font)
        self.lblSpO2 = QLabel("SpO₂: ")
        self.lblSpO2.setFont(font)

        # 性别、年龄、情绪、种族
        self.lblGender = QLabel("性别: ")
        self.lblGender.setFont(font)
        self.lblAge = QLabel("年龄: ")
        self.lblAge.setFont(font)
        self.lblEmotion = QLabel("情绪: ")
        self.lblEmotion.setFont(font)
        self.lblRace = QLabel("种族: ")
        self.lblRace.setFont(font)

        info_layout.addWidget(self.lblHR)
        info_layout.addWidget(self.lblHR2)
        info_layout.addWidget(self.lblRR)
        info_layout.addWidget(self.lblHRV)
        info_layout.addWidget(self.lblSpO2)
        info_layout.addWidget(self.lblGender)
        info_layout.addWidget(self.lblAge)
        info_layout.addWidget(self.lblEmotion)
        info_layout.addWidget(self.lblRace)
        info_layout.addStretch(1)
        
        self.signal_Plt = pg.PlotWidget()
        self.signal_Plt.setLabel('bottom', "Signal") 
        self.fft_Plt = pg.PlotWidget()
        self.fft_Plt.setLabel('bottom', "FFT") 
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.signal_Plt)
        plot_layout.addWidget(self.fft_Plt)
        right_layout = QVBoxLayout()
        right_layout.addLayout(info_layout, 1)
        right_layout.addLayout(plot_layout, 3)
        
        main_layout.addLayout(control_layout, 1, 0, 1, 3)
        main_layout.addWidget(self.lblDisplay, 0, 0, 1, 2)
        main_layout.addWidget(self.lblROI, 0, 2)
        main_layout.addLayout(right_layout, 0, 3)
        main_layout.setRowStretch(0, 8)
        main_layout.setRowStretch(1, 1)
        main_layout.setColumnStretch(0, 4)
        main_layout.setColumnStretch(1, 2)
        main_layout.setColumnStretch(2, 1)
        main_layout.setColumnStretch(3, 2)
        
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(200)
        
        self.statusBar = QStatusBar()
        self.statusBar.setFont(font)
        self.setStatusBar(self.statusBar)
        self.setGeometry(100, 100, 1280, 720)
        self.setWindowTitle("心率监测系统")
        self.show()
        
    def update(self):
        self.signal_Plt.clear()
        self.signal_Plt.plot(self.process.samples[20:], pen='g')
        self.fft_Plt.clear()
        self.fft_Plt.plot(np.column_stack((self.process.freqs, self.process.fft)), pen='g')
        
    def closeEvent(self, event):
        reply = QMessageBox.question(self, "提示", "确定要退出吗？",
            QMessageBox.Yes|QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            event.accept()
            self.input.stop()
            self.deepface_thread.stop()
            self.terminate = True
            sys.exit()
        else: 
            event.ignore()
    
    def selectInput(self):
        self.reset()
        if self.cbbInput.currentIndex() == 0:
            self.input = self.webcam
            print("Input: webcam")
            self.btnOpen.setEnabled(False)
        elif self.cbbInput.currentIndex() == 1:
            self.input = self.video
            print("Input: video")
            self.btnOpen.setEnabled(True)   
    
    def key_handler(self):
        self.pressed = waitKey(1) & 255
        if self.pressed == 27:
            print("[INFO] 退出程序")
            self.webcam.stop()
            sys.exit()
    
    def openFileDialog(self):
        self.dirname = QFileDialog.getOpenFileName(self, '打开文件')[0]
    
    def reset(self):
        self.process.reset()
        self.lblDisplay.clear()
        self.lblDisplay.setStyleSheet("background-color: #000000")
        self.rr = 0
        self.hrv = 0
        self.spo2 = 0
        self.hr_history = []
        self.red_signal, self.blue_signal = [], []
        self.bpm_history, self.rr_history, self.spo2_history = [], [], []

    def calculate_physiological_indices(self):
        pass  # 保持原逻辑

    def update_deepface_result(self, result):
        self.age = result.get('age', None)
        self.gender = result.get('dominant_gender', None)
        self.emotion = result.get('dominant_emotion', None)
        self.race = result.get('dominant_race', None)

    def main_loop(self):
        frame = self.input.get_frame()
        self.process.frame_in = frame
        if not self.terminate:
            ret = self.process.run()
        if ret:
            self.frame = self.process.frame_out
            self.f_fr = self.process.frame_ROI
            self.bpm = self.process.bpm
        else:
            self.frame = frame
            self.f_fr = np.zeros((10,10,3),np.uint8)
            self.bpm = 0
        
        if self.f_fr is not None and self.f_fr.size>0:
            roi_mean = np.mean(self.f_fr, axis=(0,1))
            b,g,r = roi_mean
            self.red_signal.append(r)
            self.blue_signal.append(b)
        
        self.calculate_physiological_indices()
        
        if self.frame is not None and self.frame.size>0:
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
            img = QImage(self.frame, self.frame.shape[1], self.frame.shape[0], 
                        self.frame.strides[0], QImage.Format_RGB888)
            self.lblDisplay.setPixmap(QPixmap.fromImage(img))
            
            # 每 30 帧发送一次给 DeepFace 线程
            self.deepface_counter += 1
            if self.deepface_counter % 30 == 0:
                self.deepface_thread.add_frame(self.frame.copy())
        
        if self.f_fr is not None and self.f_fr.size>0:
            self.f_fr = cv2.cvtColor(self.f_fr, cv2.COLOR_RGB2BGR)
            f_img = QImage(self.f_fr, self.f_fr.shape[1], self.f_fr.shape[0],
                        self.f_fr.strides[0], QImage.Format_RGB888)
            self.lblROI.setPixmap(QPixmap.fromImage(f_img))
        
        self.lblHR.setText(f"频率: {self.bpm:.2f}")
        if len(self.process.bpms) > 50:
            if max(self.process.bpms - np.mean(self.process.bpms)) < 5:
                self.lblHR2.setText(f"心率: {np.mean(self.process.bpms):.2f} bpm")
        
        self.lblRR.setText(f"呼吸率: {self.rr:.1f} bpm")
        self.lblHRV.setText(f"心率变异性: {self.hrv:.1f} ms")
        self.lblSpO2.setText(f"血氧饱和度: {self.spo2:.1f} %")

        if self.gender:
            self.lblGender.setText(f"性别: {self.gender}")
        if self.age:
            self.lblAge.setText(f"年龄: {self.age}")
        if self.emotion:
            self.lblEmotion.setText(f"情绪: {self.emotion}")
        if self.race:
            self.lblRace.setText(f"种族: {self.race}")

        self.key_handler()

    def run(self, input=None):
        print("运行中")
        self.reset()
        input = self.input
        self.input.dirname = self.dirname
        if self.input.dirname == "" and self.input==self.video:
            print("请先选择一个视频文件")
            return
        if not self.status:
            self.status = True
            input.start()
            self.btnStart.setText("停止")
            self.cbbInput.setEnabled(False)
            self.btnOpen.setEnabled(False)
            self.lblHR2.clear()
            while self.status:
                self.main_loop()
        else:
            self.status = False
            input.stop()
            self.btnStart.setText("开始")
            self.cbbInput.setEnabled(True)
            if self.cbbInput.currentIndex()==1:
                self.btnOpen.setEnabled(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GUI()
    sys.exit(app.exec_())
