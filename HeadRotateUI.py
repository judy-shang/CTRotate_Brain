import logging
import os
import sys
import shutil
import numpy as np
import pandas as pd
import pydicom as dicom
from pathlib import Path
from datetime import datetime
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QCheckBox, QLabel, 
                             QLineEdit, QFileDialog, QGroupBox, QScrollArea,
                             QMessageBox, QProgressBar, QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from HeadRotate_FalxCerebri import FalxCerebriCorrector
from HeadRotate_Affine import AffineCorrector
from pydicom.uid import ExplicitVRLittleEndian
import pydicom
import cv2
from skimage.metrics import structural_similarity as ssim
#from operator import itemgetterpip
#from skimage.metrics import niqe

class CorrectionWorker(QThread):
    """后台执行矫正任务的线程"""
    progress = pyqtSignal(int)
    log_message = pyqtSignal(str)
    finished = pyqtSignal(dict)
    
    def __init__(self, src_path, dst_path, methods, parent=None):
        super().__init__(parent)
        self.src_path = src_path
        self.dst_path = dst_path
        self.methods = methods
        self.results = {}
        self.level = logging.INFO
        self.FalxCerebri_corrector = FalxCerebriCorrector()
        self.affine_corrector = AffineCorrector()

    def load_dicom_series(self, directory):
        """加载DICOM系列图像"""
        dicom_files = [pydicom.dcmread(os.path.join(directory, f)) for f in os.listdir(directory) if f.endswith('.dcm')]
        dicom_files.sort(key=lambda x: int(x.InstanceNumber))
        return dicom_files  

    def correct_skew(self, image, angle, tx = 0, ty = 0):
        """根据角度校正图像"""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # 构造旋转矩阵
        scale = 1.0
        M_rot = cv2.getRotationMatrix2D(center, angle, scale)

        # 添加平移
        M_rot[0, 2] += tx
        M_rot[1, 2] += ty

        # 执行仿射变换
        rotated_img = cv2.warpAffine(image, M_rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) #INTER_LINEAR
        return rotated_img

    def save_corrected_dicom(self,original_dicom, corrected_image, output_dir, index, info = ''):
        """保存校正后的DICOM文件"""
        # 创建新的DICOM对象
        new_dicom = original_dicom.copy()

        # 更新像素数据
        new_dicom.PixelData = corrected_image.tobytes()

        # 更新必要的DICOM标签
        new_dicom.Rows, new_dicom.Columns = corrected_image.shape
        new_dicom.BitsStored = 16  # 根据实际数据调整
        new_dicom.HighBit = 15      # 根据实际数据调整
        new_dicom.PixelRepresentation = 1  # 无符号整数

        # 设置传输语法（确保兼容性）
        new_dicom.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        # 保存文件
        if info != '':
            output_path = os.path.join(output_dir, f"corrected_{info}_{index:03d}.dcm")
        else:
            output_path = os.path.join(output_dir, f"corrected_{index:03d}.dcm")
        new_dicom.save_as(output_path)

    def apply_rotation_and_save(self, dcm_path, output_dir, angle, tx = 0, info = ''):
        # 加载DICOM文件
        dicom_files = self.load_dicom_series(dcm_path)
        # 旋转所有DICOM图像
        for i, dicom  in enumerate(dicom_files):
            # 获取像素数据
            image = dicom.pixel_array
            # 校正图像
            corrected_image = self.correct_skew(image, angle, tx)  # 负角度以校正===椭圆角度是以x轴为起始

            # 保存校正后的DICOM文件
            self.save_corrected_dicom(dicom, corrected_image, output_dir, i)

    def run(self):
        try:
            # 获取所有患者文件夹
            patient_folders = []
            for item in os.listdir(self.src_path):
                item_path = os.path.join(self.src_path, item)
                if os.path.isdir(item_path):
                    patient_folders.append(item)
            
            total_patients = len(patient_folders)
            self.log_message.emit(f"找到 {total_patients} 个患者文件夹")
            
            results = {}
            visable = False # todo set true, to get details
            for i, patient in enumerate(patient_folders):
                self.log_message.emit(f"处理患者: {patient}")
                patient_path = os.path.join(self.src_path, patient)
                
                # 查找DICOM文件并分类薄层和厚层
                thin_files, thick_files = self.find_and_classify_dicom_files(patient_path)
                
                if not thin_files and not thick_files:
                    self.log_message.emit(f"在 {patient} 中未找到DICOM文件")
                    continue
                
                # 记录薄层和厚层文件
                patient_results = {
                    "薄层文件": thin_files,
                    "厚层文件": thick_files
                }
                
                # 处理薄层文件获取矫正角度
                if thin_files:
                    # 使用第一个薄层文件计算角度（假设同一患者的所有薄层文件角度相同）
                    sample_thin_file_dir = thin_files[0]['path']
                    slice_thickness = thin_files[0]['thickness']
                    sample_thick_file_dir = thick_files[0]['path']
                    # 应用选定的矫正方法获取角度
                    angle_results = {}
                    costTime = {}
                    ssimScore = {}
                    elapsed_time = 0
                    if 'FalxCerebri_Affine' in self.methods:
                        self.methods = list(dict.fromkeys(self.methods))
                    bone_slice_number = -1
                    for method in self.methods:
                        try:
                             # 创建输出目录
                            output_dir = os.path.join(self.dst_path, method, patient)
                            os.makedirs(output_dir, exist_ok=True)
                            # 根据方法调用相应的矫正函数
                            if method == "FalxCerebri":
                                start_time = time.time()
                                angle, bone_slice_number = self.FalxCerebri_corrector.correct_image(
                                        sample_thin_file_dir,
                                        visualize=visable,
                                        output_dir=output_dir
                                )
                                end_time = time.time()
                                elapsed_time = end_time - start_time
                                self.log_message.emit(f"大脑镰程序耗时: {elapsed_time:.4f} 秒")

                            elif method == "Affine":
                                start_time = time.time()
                                angle = self.affine_corrector.correct_image(sample_thin_file_dir,
                                        visualize=visable,
                                        output_dir=output_dir
                                )
                                end_time = time.time()
                                elapsed_time = end_time - start_time
                                self.log_message.emit(f"Affine程序耗时: {elapsed_time:.4f} 秒")
                            
                            else:
                                angle = 0
                            
                            angle_results[method] = angle
                            costTime[method] = elapsed_time

                        except Exception as e:
                            self.log_message.emit(f"方法 {method} 处理失败: {str(e)}")
                            angle_results[method] = 0

                    if 'FalxCerebri_Affine' in self.methods:
                        try: # combine
                            start_time = time.time()
                            affine_angle = self.affine_corrector.correct_image(sample_thin_file_dir,
                                        visualize=visable,
                                        output_dir=output_dir
                            )
                            angle, bone_slice_number = self.FalxCerebri_corrector.correct_image(
                                        sample_thin_file_dir,
                                        visualize=visable,
                                        output_dir=output_dir,
                                        angle_set = affine_angle
                            )
                            end_time = time.time()
                            elapsed_time = end_time - start_time
                            self.log_message.emit(f"FalxCerebri_Affine程序耗时: {elapsed_time:.4f} 秒")
                            angle_results['FalxCerebri_Affine'] = angle
                            
                            costTime['FalxCerebri_Affine'] = elapsed_time
                            self.log_message.emit(f"evaluate combine for FalxCerebri_Affine 耗时: {elapsed_time:.4f} 秒")
                        except Exception as e:
                            self.log_message.emit(f"方法 FalxCerebri_Affine 处理失败: {str(e)}")
                            angle_results[method] = 0
                    # 记录角度结果
                    patient_results["薄层层厚"] = slice_thickness
                    patient_results["矫正角度"] = angle_results
                    patient_results["耗时"] = costTime
                    # 应用相同的角度矫正所有文件（薄层和厚层）
                    if 1:
                        all_files = thin_files + thick_files
                        for file_info in all_files:
                            file_path = file_info['path']
                            for method in self.methods:
                                angle = angle_results[method]
                                if angle != 0:
                                    # 创建输出目录
                                    output_dir = os.path.join(self.dst_path, method, patient)
                                    os.makedirs(output_dir, exist_ok=True)
                                    # 应用旋转矫正并保存（这里只是复制原文件作为示例）
                                    # 获取上级目录路径
                                    parent_dir_name = os.path.basename(file_path)
                                    output_path = os.path.join(output_dir, parent_dir_name)
                                    os.makedirs(output_path, exist_ok=True)
                                    
                                    self.apply_rotation_and_save(file_path, output_path, angle)  # 应用旋转, 保存旋转后的图像
                                    self.log_message.emit(f"{method} get angle: {angle} for {file_path}, rotate and saved now")
                
                # 记录患者结果
                results[patient] = patient_results
                
                # 更新进度
                progress = int((i + 1) / total_patients * 100)
                self.progress.emit(progress)
            
            # 保存结果到Excel
            self.save_results_to_excel(results)
            self.finished.emit(results)
            
        except Exception as e:
            self.log_message.emit(f"处理过程中发生错误: {str(e)}")
            self.finished.emit({})
    
    def find_and_classify_dicom_files(self, folder_path):
        """在文件夹中查找并分类DICOM文件（薄层和厚层）"""
        thin_files = []  # 薄层文件（≤2mm）
        thick_files = []  # 厚层文件（>2mm）
        
        for root, _, files in os.walk(folder_path):
            if len(files) < 10:
                continue # 跳过定位文件
            for file in files:
                if file.lower().endswith(('.dcm', '.dicom')):
                    file_path = os.path.join(root, file)
                    try:
                        ds = dicom.dcmread(file_path)
                        # skip uAI result
                        station_name = getattr(ds, 'StationName')
                        if (station_name == 'UII_AI'):
                            break
                        slice_thickness = float(getattr(ds, 'SliceThickness', 0))
                        
                        file_info = {
                            'path': os.path.dirname(file_path), # 获取厚薄层路径
                            'thickness': slice_thickness
                        }
                        
                        if slice_thickness <= 2.5 and slice_thickness > 0.1:  # 假设≤2mm为薄层
                            thin_files.append(file_info)
                            break
                        else:
                            thick_files.append(file_info)
                            break
                            
                    except Exception as e:
                        self.log_message.emit(f"读取DICOM文件失败 {file_path}: {str(e)}")
        
        return thin_files, thick_files
    
    def save_results_to_excel(self, results):
        """将结果保存到Excel文件"""
        try:
            # 准备数据
            data = []
            for patient, patient_info in results.items():
                if "矫正角度" in patient_info:  # 只有有薄层数据的患者才记录
                    row = {
                        "文件路径": patient,
                        "薄层层厚": patient_info.get("薄层层厚", "N/A"),
                    }
                    
                    # 添加每种方法的矫正角度
                    for method, angle in patient_info.get("矫正角度", {}).items():
                        row[method + "角度"] = angle
                    for method, costTime in patient_info.get("耗时", {}).items():
                        row[method + "耗时"] = costTime
                    data.append(row)
            
            # 创建DataFrame并保存
            if data:
                df = pd.DataFrame(data)
                excel_path = os.path.join(self.dst_path, "矫正结果.xlsx")
                df.to_excel(excel_path, index=False)
                self.log_message.emit(f"结果已保存到: {excel_path}")
            else:
                self.log_message.emit("没有找到薄层数据，未生成Excel文件")
            
        except Exception as e:
            self.log_message.emit(f"保存Excel失败: {str(e)}")



class HeadCorrectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger('HeadRotateUI')
        self.logger.setLevel(logging.INFO)  # 设置日志级别

        # 2. 创建 FileHandler
        log_filename = datetime.now().strftime(r'.\HeadRotateUI__output_%Y-%m-%d.log')
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')

        # 3. 创建 formatter 并设置给 handler
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)

        # 4. 将 handler 添加到 logger
        self.logger.addHandler(file_handler)
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('头颅CT图像矫正工具')
        self.setGeometry(100, 100, 800, 600)
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        layout = QVBoxLayout(central_widget)
        
        # 源路径选择
        src_group = QGroupBox("源文件夹")
        src_layout = QHBoxLayout()
        self.src_edit = QLineEdit()
        src_btn = QPushButton("浏览")
        src_btn.clicked.connect(self.select_src)
        src_layout.addWidget(QLabel("DICOM图像文件夹:"))
        src_layout.addWidget(self.src_edit)
        src_layout.addWidget(src_btn)
        src_group.setLayout(src_layout)
        layout.addWidget(src_group)
        
        # 目标路径选择
        dst_group = QGroupBox("目标文件夹")
        dst_layout = QHBoxLayout()
        self.dst_edit = QLineEdit()
        dst_btn = QPushButton("浏览")
        dst_btn.clicked.connect(self.select_dst)
        dst_layout.addWidget(QLabel("矫正图像存储文件夹:"))
        dst_layout.addWidget(self.dst_edit)
        dst_layout.addWidget(dst_btn)
        dst_group.setLayout(dst_layout)
        layout.addWidget(dst_group)
        
        # 方法选择
        methods_group = QGroupBox("矫正方法")
        methods_layout = QVBoxLayout()
        self.methods = {
            "FalxCerebri": QCheckBox("大脑镰"),
            "Affine": QCheckBox("Affine配准"),
            "FalxCerebri_Affine": QCheckBox("大脑镰+Affine配准")
        }
        for method in self.methods.values():
            methods_layout.addWidget(method)
        methods_group.setLayout(methods_layout)
        layout.addWidget(methods_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # 日志窗口
        log_group = QGroupBox("处理日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        # 开始按钮
        self.start_btn = QPushButton("开始矫正")
        self.start_btn.clicked.connect(self.start_correction)
        layout.addWidget(self.start_btn)
        
        # 初始化工作线程
        self.worker = None
        
    def select_src(self):
        path = QFileDialog.getExistingDirectory(self, "选择DICOM图像文件夹")
        if path:
            self.src_edit.setText(path)
            
    def select_dst(self):
        path = QFileDialog.getExistingDirectory(self, "选择矫正图像存储文件夹")
        if path:
            self.dst_edit.setText(path)
            
    def log_message(self, message):
        """添加消息到日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.logger.info(message)
        
    def start_correction(self):
        """开始矫正过程"""
        src_path = self.src_edit.text()
        dst_path = self.dst_edit.text()
        
        if not src_path or not dst_path:
            QMessageBox.warning(self, "警告", "请先选择源文件夹和目标文件夹")
            return
            
        # 获取选中的方法
        selected_methods = []
        for name, checkbox in self.methods.items():
            if checkbox.isChecked():
                selected_methods.append(name)
                
        if not selected_methods:
            QMessageBox.warning(self, "警告", "请至少选择一种矫正方法")
            return
            
        # 创建并启动工作线程
        self.worker = CorrectionWorker(src_path, dst_path, selected_methods)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log_message.connect(self.log_message)
        self.worker.finished.connect(self.on_finished)


        self.worker.start()
        
        # 禁用开始按钮
        self.start_btn.setEnabled(False)
        self.log_message("开始处理...")
        
    def on_finished(self, results):
        """处理完成后的回调"""
        self.start_btn.setEnabled(True)
        self.log_message("处理完成!")
        
        # 显示结果统计
        patient_count = len(results)
        thin_count = sum(1 for info in results.values() if "薄层层厚" in info)
        thick_count = sum(len(info.get("厚层文件", [])) for info in results.values())
        self.log_message(f"共处理 {patient_count} 个患者, {thin_count} 个薄层文件, {thick_count} 个厚层文件")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = HeadCorrectionApp()
    window.show()
    sys.exit(app.exec_())