import numpy as np
from scipy import ndimage
from skimage import morphology, measure
import matplotlib.pyplot as plt

class BinaryBettiCalculator:
    def __init__(self):
        self.image = None
        self.skeleton = None
        
    def set_image(self, binary_image):
        """
        设置二值图像
        
        参数:
        - binary_image: 0-1二值图像，numpy数组类型
        """
        # 确保输入是二值图像
        unique_values = np.unique(binary_image)
        if not np.array_equal(unique_values, [0, 1]) and not np.array_equal(unique_values, [0]) and not np.array_equal(unique_values, [1]):
            raise ValueError("输入必须是0-1二值图像")
        
        if len(binary_image.shape) == 4:
            binary_image = binary_image[0]
            
        if type(binary_image) != np.ndarray:
            binary_image = binary_image.numpy()
        self.image = binary_image.astype(bool)  # 转换为布尔类型以提高效率
        return self.image

    def clean_noise(self, min_size=64):
        """
        可选的噪声清理步骤
        
        参数:
        - min_size: 最小连通区域大小，小于此大小的区域将被移除
        """
        if self.image is None:
            raise ValueError("请先设置输入图像")
            
        # 移除小连通区域
        cleaned = morphology.remove_small_objects(self.image, min_size=min_size)
        # 填充小孔洞
        cleaned = morphology.remove_small_holes(cleaned, min_size=min_size)
        
        self.image = cleaned
        return cleaned

    def compute_beta0(self):
        """
        计算β₀（连通分量数）
        """
        if self.image is None:
            raise ValueError("请先设置输入图像")
            
        # 使用3D连通分量标记
        labeled, num_components = ndimage.label(self.image)
        return num_components

    def compute_beta2(self):
        """
        计算β₂（空腔数量）
        """
        if self.image is None:
            raise ValueError("请先设置输入图像")
            
        # 填充所有封闭空腔
        filled = ndimage.binary_fill_holes(self.image)
        
        # 找到内部空腔（填充图像与原图的差异）
        cavities = filled ^ self.image
        
        # 标记独立空腔
        labeled_cavities, num_cavities = ndimage.label(cavities)
        
        return num_cavities

    def compute_beta1(self):
        """
        计算β₁（环的数量）
        使用欧拉特征计算
        """
        if self.image is None:
            raise ValueError("请先设置输入图像")
            
        # 计算骨架用于可视化
        self.skeleton = morphology.skeletonize_3d(self.image)
        
        # 计算欧拉特征
        euler_number = measure.euler_number(self.image)
        
        # 使用欧拉公式：χ = β₀ - β₁ + β₂
        beta0 = self.compute_beta0()
        beta2 = self.compute_beta2()
        beta1 = beta0 - euler_number + beta2
        
        return beta1

    def compute_all_betti(self, binary_image, clean=False, min_size=64):
        """
        计算所有Betti数
        
        参数:
        - binary_image: 0-1二值图像
        - clean: 是否进行噪声清理
        - min_size: 噪声清理的最小区域大小
        
        返回:
        - (β₀, β₁, β₂)
        """
        # 设置图像
        self.set_image(binary_image)
        
        # 可选的噪声清理
        if clean:
            self.clean_noise(min_size)
        
        # 计算所有Betti数
        beta0 = self.compute_beta0()
        beta2 = self.compute_beta2()
        beta1 = self.compute_beta1()
        
        return beta0, beta1, beta2

def example_usage():
    """
    使用示例
    """
    # 创建一个示例3D二值图像
    size = 50
    image = np.zeros((size, size, size), dtype=np.int8)
    
    # 添加一个球体（值为1）
    x, y, z = np.ogrid[-size/2:size/2, -size/2:size/2, -size/2:size/2]
    sphere = x**2 + y**2 + z**2 <= (size/4)**2
    image[sphere] = 1
    
    # 添加一个内部空腔（值为0）
    small_sphere = x**2 + y**2 + z**2 <= (size/8)**2
    image[small_sphere] = 0
    
    # 创建计算器实例
    calculator = BinaryBettiCalculator()
    
    # 计算Betti数
    beta0, beta1, beta2 = calculator.compute_all_betti(image)
    
    # 打印结果
    print("Betti数计算结果:")
    print(f"β₀ (连通分量数): {beta0}")
    print(f"β₁ (环的数量): {beta1}")
    print(f"β₂ (空腔数量): {beta2}")


# 实际使用示例
if __name__ == "__main__":
    example_usage()