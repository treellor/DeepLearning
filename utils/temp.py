from PIL import Image
import os



def convert_to_grayscale(input_folder, output_folder):
    # 检查输出文件夹是否存在，若不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # 检查文件是否为图片
        if not os.path.isfile(input_path) or not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            continue

        # 打开图片并转换为黑白
        with Image.open(input_path) as img:
            grayscale_img = img.convert('L').convert('RGB')  # 将图片转换为黑白，'L'表示灰度图像

            # 构建输出路径
            output_filename = os.path.splitext(filename)[0] + '.jpg'
            output_path = os.path.join(output_folder, output_filename)

            # 保存黑白图片到输出文件夹
            grayscale_img.save(output_path)


if __name__ == "__main__":
    input_folder = r"D:\project\Pycharm\DeepLearning\data\face"  # 替换为您的输入文件夹路径
    output_folder = r"D:\project\Pycharm\DeepLearning\data\gray"  # 替换为您的输出文件夹路径

    convert_to_grayscale(input_folder, output_folder)
