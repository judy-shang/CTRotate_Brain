from datetime import datetime
import numpy as np
import pydicom
from sklearn.decomposition import PCA
from scipy import ndimage
import cv2
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging
import os
import ants
import SimpleITK as sitk
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import nilearn.image as nimg
from scipy.spatial.transform import Rotation as R

class AffineCorrector:
    """
    Affine头颅矫正类
    """
    
    def __init__(self):
        # 1. 创建 logger 实例
        self.logger = logging.getLogger('RotateAffine')
        self.logger.setLevel(logging.INFO)  # 设置日志级别

        # 2. 创建 FileHandler
        log_filename = datetime.now().strftime(r'.\Rotate__Affine_output_%Y-%m-%d.log')
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')

        # 3. 创建 formatter 并设置给 handler
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)

        # 4. 将 handler 添加到 logger
        self.logger.addHandler(file_handler)
        # 设置中文字体（Windows 常用字体）
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
        # 或 ['Microsoft YaHei']  # 微软雅黑
        # 或 ['KaiTi']           # 楷体
        # 或 ['FangSong']        # 仿宋

        # 解决负号显示问题（可选）
        plt.rcParams['axes.unicode_minus'] = False  

        template_path = r".\scct.nii"           # 模板NIfTI文件
        self.fixed = ants.image_read(template_path)

    def load_dicom_series(self, directory):
        """加载DICOM系列图像"""
        dicom_files = [pydicom.dcmread(os.path.join(directory, f)) for f in os.listdir(directory) if f.endswith('.dcm')]
        dicom_files.sort(key=lambda x: int(x.InstanceNumber))
        return dicom_files    
    

    def dicom_series_to_nifti(self, dicom_dir, output_nii):
        """
        将DICOM系列转换为NIfTI格式
        """
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)
        reader.SetFileNames(dicom_files)
        image = reader.Execute()
        
        sitk.WriteImage(image, output_nii)
        return output_nii

    def correct_image(self, file_path: str, 
                     visualize: bool = False,
                     output_dir: Optional[str] = None) -> Tuple[float, np.ndarray]:
        """
        完整的图像矫正流程
        
        Args:
            file_path: DICOM文件路径
            visualize: 是否生成可视化结果
            output_dir: 可视化结果输出目录
            
        Returns:
            Tuple[矫正角度]
        """
        try:
            dicomFiles = self.load_dicom_series(file_path)
            # Step 1: 转换DICOM到NIfTI
            self.logger.info("Converting DICOM to NIfTI...")
            input_nii = os.path.join(output_dir, 'input.nii')
            self.dicom_series_to_nifti(file_path, input_nii)
            
            # Step 2: 使用ANTs进行刚性配准
            self.logger.info("Performing syn registration with ANTs...")
            moving = ants.image_read(input_nii)
            
            # 刚性配准
            reg_result = ants.registration(
                fixed=self.fixed,
                moving=moving,
                type_of_transform='Affine', #Rigid
                verbose=True
            )
            
            # 保存配准后的图像
            reg_nii = os.path.join(output_dir, 'registered_Affine.nii')
            ants.image_write(reg_result['warpedmovout'], reg_nii)
            
            # 读取变换矩阵
            transform_matrix = ants.read_transform(reg_result['fwdtransforms'][0])
            self.logger.info(f"Transform matrix file: {transform_matrix}")

            params = transform_matrix.parameters  # 共 12 个参数: 3x3 旋转矩阵 + 3 个平移
            matrix = np.array(params[:9]).reshape(3, 3)
            translation = np.array(params[9:])
            
            rot = R.from_matrix(matrix)
            euler_angles_rad = rot.as_euler('zyx')  # 可以改为 'zyx' 视坐标系而定
            euler_angles_deg = np.degrees(euler_angles_rad)

            self.logger.info(f"旋转角度zyx（度）：{euler_angles_deg}")
            self.logger.info(f"平移向量xyz（mm）：{translation}")
        
            output_path = os.path.join(output_dir, 'Wrapped')
            os.makedirs(output_path, exist_ok=True)
            #self.convert_nii_2_dicom(dicomFiles[0], reg_nii, output_path)
            return euler_angles_deg[0]
            
        except Exception as e:
            self.logger.error(f"图像矫正失败: {str(e)}")
            raise
    