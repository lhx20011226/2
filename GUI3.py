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
        info_layout.addWidget(self.lblHR)
        info_layout.addWidget(self.lblHR2)
        info_layout.addWidget(self.lblRR)
        info_layout.addWidget(self.lblHRV)
        info_layout.addWidget(self.lblSpO2)
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

    # ---------- 优化后的生理指标计算 ----------
    def calculate_physiological_indices(self):
        # ---------- 心率 ----------
        if len(self.red_signal) > 150:
            signal = np.array(self.red_signal[-300:])
            signal = signal - np.mean(signal)
            b, a = butter(3, [0.8/(self.process.fps/2), 3.0/(self.process.fps/2)], btype='band')
            filtered = filtfilt(b, a, signal)
            freqs = np.fft.rfftfreq(len(filtered), d=1/self.process.fps)
            fft_mag = np.abs(np.fft.rfft(filtered))
            mask = (freqs >= 0.8) & (freqs <= 3.0)
            if np.any(mask):
                top_idx = np.argsort(fft_mag[mask])[-3:]
                top_freqs = freqs[mask][top_idx]
                top_mag = fft_mag[mask][top_idx]
                self.bpm = np.sum(top_freqs*top_mag)/np.sum(top_mag) * 60
            self.bpm_history.append(self.bpm)
            if len(self.bpm_history) > 5:
                self.bpm = np.mean(self.bpm_history[-5:])
        else:
            self.bpm = 0

        # ---------- HRV ----------
        if len(self.process.bpms) > 10:
            rr_intervals = 60.0 / np.array(self.process.bpms)
            diff_rr = np.diff(rr_intervals)
            self.hrv = np.sqrt(np.mean(diff_rr**2)) * 1000
        else:
            self.hrv = 0

        # ---------- 呼吸率 ----------
        if len(self.process.samples) > 50:
            signal = np.array(self.process.samples[-200:])
            signal = signal - np.mean(signal)
            try:
                b, a = butter(2, [0.1/(self.process.fps/2), 0.5/(self.process.fps/2)], btype='band')
                filtered = filtfilt(b, a, signal)
                freqs = np.fft.rfftfreq(len(filtered), d=1/self.process.fps)
                fft_mag = np.abs(np.fft.rfft(filtered))
                mask = (freqs >= 0.1) & (freqs <= 0.5)
                if np.any(mask):
                    peak_freq = freqs[mask][np.argmax(fft_mag[mask])]
                    rr_inst = peak_freq * 60
                else:
                    rr_inst = 0
                self.rr_history.append(rr_inst)
                if len(self.rr_history)>5:
                    self.rr = np.mean(self.rr_history[-5:])
            except:
                self.rr = 0
        else:
            self.rr = 0

        # ---------- SpO2 ----------
        if len(self.red_signal) > 150 and len(self.blue_signal) > 150:
            red = np.array(self.red_signal[-300:])
            blue = np.array(self.blue_signal[-300:])
            red = np.convolve(red, np.ones(5)/5, mode='valid')
            blue = np.convolve(blue, np.ones(5)/5, mode='valid')
            ac_red, dc_red = np.std(red), np.mean(red)
            ac_blue, dc_blue = np.std(blue), np.mean(blue)
            if dc_red>0 and dc_blue>0:
                R = (ac_red/dc_red) / (ac_blue/dc_blue)
                spo2_inst = 100 - 5*(R - 0.4)
                spo2_inst = np.clip(spo2_inst, 90, 100)
                self.spo2_history.append(spo2_inst)
                if len(self.spo2_history) > 5:
                    self.spo2 = np.mean(self.spo2_history[-5:])
            else:
                self.spo2 = 0

    # ---------- 主循环 ----------
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
            cv2.putText(self.frame, f"FPS {self.process.fps:.2f}",
                        (20, 460), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,255),2)
            img = QImage(self.frame, self.frame.shape[1], self.frame.shape[0], 
                        self.frame.strides[0], QImage.Format_RGB888)
            self.lblDisplay.setPixmap(QPixmap.fromImage(img))
        
        if self.f_fr is not None and self.f_fr.size>0:
            self.f_fr = cv2.cvtColor(self.f_fr, cv2.COLOR_RGB2BGR)
            self.f_fr = np.transpose(self.f_fr, (0,1,2)).copy()
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

        self.key_handler()

    # ---------- 运行 ----------
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
