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


class FalxCerebriCorrector:
    """
    FalxCerebriCorrector头颅矫正类
    使用大脑镰角度对头颅CT图像进行自动矫正
    """
    def __init__(self):
        # 1. 创建 logger 实例
        self.logger = logging.getLogger('RotateFalxCerebri')
        self.logger.setLevel(logging.INFO)  # 设置日志级别

        # 2. 创建 FileHandler
        log_filename = datetime.now().strftime(r'.\Rotate__FalxCerebri_output_%Y-%m-%d.log')
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')

        # 3. 创建 formatter 并设置给 handler
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)

        # 4. 将 handler 添加到 logger
        self.logger.addHandler(file_handler)
        # 设置中文字体（Windows 常用字体）
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体

        # 解决负号显示问题（可选）
        plt.rcParams['axes.unicode_minus'] = False  

    def load_dicom_series(self, directory):
        """加载DICOM系列图像"""
        dicom_files = [pydicom.dcmread(os.path.join(directory, f)) for f in os.listdir(directory) if f.endswith('.dcm')]
        dicom_files.sort(key=lambda x: int(x.InstanceNumber))
        return dicom_files    
    def get_scan_direction(self, sorted_slices):
        # 通过Z坐标变化判断
        z1 = sorted_slices[0].ImagePositionPatient[2]
        z2 = sorted_slices[1].ImagePositionPatient[2]
        if z2 > z1:
            self.logger.info("从脚到头扫描（Z坐标递增）")
            return 1
        else:
            self.logger.info("从头到脚扫描（Z坐标递减）")
            return 0
    
    def get_start_location(self, dicom_files, direction):
        # 根据Position的值进行排序，头 > 脚
        sorted_slicesNew = sorted(dicom_files, key=lambda s: s.ImagePositionPatient[2], reverse=True)

        # 定义阈值：CT值大于-200（排除空气）
        threshold = 150

        # 从最高层向下遍历
        for slice in sorted_slicesNew:
            img_array = slice.pixel_array
            # 转换为CT值（Hounsfield Unit, HU）
            if hasattr(slice, 'RescaleIntercept') and hasattr(slice, 'RescaleSlope'):
                hu_array = img_array * slice.RescaleSlope + slice.RescaleIntercept
            else:
                hu_array = img_array  # 默认未校准
            
            # 检查是否存在组织（非空气）
            if np.any(hu_array > threshold):
                return slice.ImagePositionPatient[2]  # 返回第一个符合条件的DICOM文件
            
    def get_FalxCerebri_slices(self, dicom_files, startLocation, endLocation, boneLocation):
        result_files = []
        count_n = 0
        slice_number = 0
        lastPosition = -1
        for slice in dicom_files:
            if slice.ImagePositionPatient[2] <= startLocation and slice.ImagePositionPatient[2] >= endLocation:
                result_files.append(slice)  
                count_n = count_n + 1
            if abs(slice.ImagePositionPatient[2] - boneLocation) < 3:
                if slice_number == 0 or slice.ImagePositionPatient[2] < lastPosition:
                    slice_number = slice.InstanceNumber
                    lastPosition= slice.ImagePositionPatient[2] 
        self.logger.info(f"大脑镰层数: {count_n}, slice_number: {slice_number}")
        return result_files, count_n, slice_number  

    def preprocess_slice(self, imageFile, output_dir, isVisible):
        """预处理单张CT图像"""
        # 中值滤波去噪
        image = imageFile.pixel_array
        filtered = cv2.medianBlur(image.astype(np.uint16), 3)
        
        # 转换为HU值
        intercept = imageFile.RescaleIntercept
        slope = imageFile.RescaleSlope
        img = slope * filtered + intercept
        
        # 预处理
        hu_thresh = 100
        remove_bone = np.where(img > hu_thresh, 0, img)
        #return remove_bone
        remove_lower_zero = np.where(remove_bone < 0, 0, remove_bone)
        remove_bone_copy = remove_lower_zero.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))
        remove_open = cv2.morphologyEx(remove_bone_copy, cv2.MORPH_OPEN, kernel, iterations=5)

        remove_bone_erode_masked = np.where(remove_open <= 0, 0, remove_lower_zero)
        remove_bone_erode_masked_binary= np.where(remove_bone_erode_masked <= 0, 0, 1).astype(np.uint8)
        remove_bone_erode_masked_normalized = cv2.normalize(remove_bone_erode_masked, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # 查找轮廓
        contours, _ = cv2.findContours(remove_bone_erode_masked_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建掩码（用于存储外边界）
        mask = np.zeros_like(remove_bone_erode_masked_binary)

        # 只绘制外边界（不填充）
        cv2.drawContours(mask, contours, -1, 255, thickness=3)  # `thickness` 控制删除宽度

        # 生成最终结果：用掩码擦除外边界
        result_binary = cv2.bitwise_and(remove_bone_erode_masked_binary, cv2.bitwise_not(mask))

        result_img = np.where(result_binary <= 0, 0, remove_bone_erode_masked)
        if isVisible:    
            plt.figure(figsize=(12, 6))
            plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('(a)原始图像')
            plt.subplot(132), plt.imshow(remove_lower_zero, cmap='gray'), plt.title('(b)去骨后')
            plt.subplot(133), plt.imshow(result_img, cmap='gray'), plt.title(f'(c)降噪后')
            outPath = os.path.join(output_dir, f'midline_preprocess_{imageFile.InstanceNumber}.png')
            plt.savefig(outPath, dpi=300)
            plt.close()

        return remove_bone_erode_masked

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

    def compute_skull_max_height(self, image, mask=None, angles = []):
        """
        计算二值 mask 图像中每列的非零像素 y 范围（高度），返回最大值。

        参数:
            mask (np.ndarray): 2D 二值图像（颅骨区域）

        返回:
            int: 所有列中最大的 y 高度（颅骨最大竖直高度）
        """

        h, w = image.shape
        half = w // 2
        finAngle = -180
        finHeight = 0
        finAxis = 0

        finWidth = w

        for angle in angles:
            max_height = 0
            max_width = 0
            axis = half
            tmpImage = image.copy()
            if mask is not None:
                tmpImage = np.where(mask, tmpImage, 0)

            corrected_image = self.correct_skew(tmpImage, angle)
            #plt.figure(figsize=(12, 8))
        
            # 原始图像和对称性
            #plt.imshow(corrected_image, cmap='gray')
            #plt.show()
            for x in range(w):
                column = corrected_image[:, x]
                y_indices = np.where(column > 0)[0]
                if len(y_indices) > 0:
                    height_this_column = y_indices[-1] - y_indices[0] + 1
                    if max_height < height_this_column:
                        max_height = height_this_column
                        axis = x
            for y in range(h):
                row = corrected_image[y, :]
                x_indices = np.where(row > 0)[0]
                if len(x_indices) > 0:
                    width_this_row = x_indices[-1] - x_indices[0] + 1
                    if max_width < width_this_row:
                        max_width = width_this_row
            self.logger.info(f"compute_skull_max_height get from angles: {angle}, max_height: {max_height}, max_width: {max_width} now ")  

            if max_height >= finHeight and max_width <= finWidth:
                finHeight = max_height
                finAngle = angle
                finAxis = axis
                finWidth = max_width
            elif (max_height - finHeight +  finWidth - max_width) > 0:
                finHeight = max_height
                finAngle = angle
                finAxis = axis
                finWidth = max_width

        self.logger.info(f"compute_skull_max_height get final angle from angles is: {finAngle}, finHeight: {finHeight}, finAxis: {finAxis}, finWidth: {finWidth}")     
        return finAngle, finAxis

    def detect_midline(self, combined, df, outSubDirPath, testImage, isVisible, angle_set = None):
        """检测中线并返回旋转角度"""
        combined_norm = np.where(combined <= 0, 0, combined)
        combined_normal = cv2.normalize(combined_norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # 定义旋转角度范围
        angles = np.arange(-60, 60, 1)  # 从 -90 度到 90 度，步长为 1 度

        # 初始化最大累计值和对应的角度
        max_sum = 100
        best_angle = 0
        # 只保留 y 轴（中心）附近的列，比如中心 ±w//4
        filtered_img = combined_normal.copy()
        (h, w) = filtered_img.shape[:2]
        center = (w // 2, h // 2)
        center_x = w // 2  # 图像中心
        width_range = w // 8  # 你可以调整这个值来扩大或缩小保留区域
        center_index = width_range
        # 假设 filtered_column_sums_list 是多个图像的累计值列表
        filtered_column_sums_list = []
        if 0:
            # 预旋转，mse确定方向
            angle1= 10
            angle2 = -10
            # 校正图像
            image_2d = testImage.pixel_array
            intercept = testImage.RescaleIntercept
            slope = testImage.RescaleSlope
            img = image_2d * slope + intercept
            
            # 预处理
            hu_thresh = 100
            bone = np.where(img > hu_thresh, img, 0)

            better_angle, axis = self.compute_skull_max_height(bone, angles=[angle1, angle2])

            if better_angle == angle1:
                subAngles = np.arange(0, 60, 1)  # 从 -30 度到 30 度，步长为 1 度 # 大部分是在-30 ~30度
            else:
                subAngles = np.arange(-60, 0, 1)  # 从 -30 度到 30 度，步长为 1 度 # 大部分是在-30 ~30度
        if angle_set:
            angle_set_int = int(angle_set)
            subAngles = np.arange(angle_set_int - 5, angle_set_int + 5, 1)
        else:
            subAngles = np.arange(-45, 45, 1) 
        # 遍历每个角度
        for angle in subAngles:
            # 获取旋转矩阵
            filtered_img = combined_normal.copy()
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            # 旋转图像
            rotated_image = cv2.warpAffine(filtered_img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
            # 计算垂直方向的像素累计值
            column_sums = np.sum(rotated_image, axis=0)
            # 选择中心区域的 column_sums
            filtered_column_sums = column_sums[center_x - width_range : center_x + width_range]
            # 找到累计值的最大值
            current_max_sum = np.max(filtered_column_sums)
            # 获取最大值的索引
            max_index = abs(np.argmax(filtered_column_sums) - width_range)
            # 更新最大累计值和对应的角度
            if current_max_sum > max_sum:
                max_sum = current_max_sum
                best_angle = angle
                center_index = max_index # 计算中心索引
        diff = max_sum*0.01            
        subAngles2 = subAngles   # 从 -30 度到 30 度，步长为 1 度 # 大部分是在-30 ~30度
        # 遍历每个角度
        for angle in subAngles2:
            # 获取旋转矩阵
            filtered_img = combined_normal.copy()
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            # 旋转图像
            rotated_image = cv2.warpAffine(filtered_img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
            # 计算垂直方向的像素累计值
            column_sums = np.sum(rotated_image, axis=0)
            # 选择中心区域的 column_sums
            filtered_column_sums = column_sums[center_x - width_range : center_x + width_range]
            # 找到累计值的最大值
            current_max_sum = np.max(filtered_column_sums)
            # 获取最大值的索引
            max_index = abs(np.argmax(filtered_column_sums) - width_range)
            # 更新最大累计值和对应的角度
            if abs(np.int64(current_max_sum) - np.int64(max_sum)) < diff:
                sum_result = {"angle": angle, "max_sum": current_max_sum, "data": filtered_column_sums, "center": max_index}
                filtered_column_sums_list.append(sum_result)

        # 二次矫正       
        for item in filtered_column_sums_list: # 可能存在多个最大值接近的数据，选择相差不大，而且最接近center的
            if abs(np.int64(item["max_sum"]) - np.int64(max_sum)) < diff and item["center"] < center_index and \
                (abs(np.int64(item["max_sum"]) - np.int64(max_sum))/np.int64(max_sum)) < (abs(item["center"]  - center_index) / width_range):
                self.logger.critical(f"old rotate angle: angle is{best_angle}, max_sum is {max_sum}, center_index is {center_index} change to \
                                    angle is {item['angle']}, max_sum is {item['max_sum']}, center_index is {item['center']}")
                best_angle = item["angle"]
                max_sum = item["max_sum"]
                center_index = item["center"]
                
        if isVisible:  
        
            filtered_img = combined_normal.copy()
            (h, w) = filtered_img.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, best_angle, 1.0)

            # 旋转图像
            rotated_image = cv2.warpAffine(filtered_img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
                
            plt.figure(figsize=(12, 6))
            plt.subplot(121), plt.imshow(combined_normal, cmap='gray'), plt.title('(d)多层脑组织叠加图')
            #plt.subplot(222), plt.imshow(filtered_img, cmap='gray'), plt.title('filtered_img Image')
            #plt.subplot(223), plt.imshow(combined_removed, cmap='gray'), plt.title('combined_removed Image')
            plt.subplot(122), plt.imshow(rotated_image, cmap='gray'), plt.title(f'(e)角度矫正结果图')
            outPath = os.path.join(outSubDirPath, f"midline_detection_rotated_result.png")
            plt.savefig(outPath, dpi=300)
            #plt.show()
            plt.close()

            plt.figure(figsize=(10, 5))
            for item in filtered_column_sums_list:
                plt.plot(item["data"], label=f"旋转角度：{item["angle"]}, 叠加值：{item["max_sum"]}, 极值点：{item["center"]}")  # 绘制每张图像的曲线

            plt.xlabel("X轴位置(起止点为原图的3/8宽度至5/8宽度)")
            plt.ylabel("叠加灰度值")
            plt.title("(f)不同旋转角度下，X轴的叠加灰度曲线图")
            plt.legend()  # 添加图例
            plt.grid(True)
            outPath = os.path.join(outSubDirPath, f"midline_detection_rotated_detailInfo.png")
            plt.savefig(outPath, dpi=300)
            #plt.show()
            plt.close()


        # 输出最佳角度
        self.logger.info(f"最佳角度: {best_angle} 度")
        #print(f"最佳角度: {best_angle} 度")
        return best_angle



    def correct_image(self, file_path: str, 
                     visualize: bool = False,
                     output_dir: Optional[str] = None,
                     angle_set = None) -> Tuple[float, np.ndarray]:
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
            # 1. 加载图像, 确定层面
            dicom_files = self.load_dicom_series(file_path)
            direction = self.get_scan_direction(dicom_files) # 1 从脚到头扫描; 0 从头到脚扫描
            startLocation = self.get_start_location(dicom_files, direction)
            # 颅顶距大脑镰起始位置约20mm
            brainFalxStartLocation = startLocation - 25
            # 大脑镰范围约20mm
            brainFalxEndLocation = brainFalxStartLocation - 15
            self.logger.info(f"大脑镰起始位置: {brainFalxStartLocation}, 结束位置: {brainFalxEndLocation}")
            brainFalxBoneLocation = brainFalxEndLocation 
            top_slices_files, count_n, bone_slice_number = self.get_FalxCerebri_slices(dicom_files, brainFalxStartLocation, brainFalxEndLocation, brainFalxBoneLocation)  # 取最后top_n层（颅顶）
            top_slices = [s.pixel_array for s in top_slices_files]
            # 预处理并叠加
            processed = [self.preprocess_slice(s, output_dir, visualize) for s in top_slices_files]
            combined = np.sum(processed, axis=0) / count_n
            
            # 检测旋转角度
            size = len(dicom_files)
            rotateAngle = self.detect_midline(combined, top_slices_files[0], output_dir, dicom_files[bone_slice_number], visualize, angle_set)
            self.logger.info(f"Detected rotation angle: {rotateAngle:.2f} degrees, bone_slice_number: {bone_slice_number}")
            return rotateAngle,bone_slice_number
            
        except Exception as e:
            self.logger.error(f"图像矫正失败: {str(e)}")
            raise
    