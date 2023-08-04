import os
from PIL import Image

# 输入文件夹和输出文件夹的路径
input_folder = r"D:\project\Pycharm\DeepLearning\data\flag"
output_folder = r"D:\project\Pycharm\DeepLearning\data\flag128"

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有图像文件
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # 只处理jpg和png格式的图像
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 读取图像
        image = Image.open(input_path)

        # 缩放图像到原来的一半大小
        width, height = image.size
        new_width = width // 4
        new_height = height // 4
        scaled_image = image.resize((new_width, new_height))

        # 保存缩放后的图像
        scaled_image.save(output_path)

print("图像缩放并保存完成！")
