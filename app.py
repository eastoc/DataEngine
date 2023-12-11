import gradio as gr
import os
from PIL import Image

def load_images(folder_path):
    # 获取给定文件夹路径下的所有文件
    files = os.listdir(folder_path)
    # 过滤出图片文件
    images = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    return images

def show_image(folder_path, image_name):
    # 根据图片名称和文件夹路径显示图片
    image_path = os.path.join(folder_path, image_name)
    return Image.open(image_path)

# 创建一个文件夹输入组件
folder_input = gr.Textbox(label="图像文件夹路径")

# 创建一个用于显示图片名称的组件
image_dropdown = gr.Dropdown(label="图片列表")

# 创建一个用于显示图片的组件
image_output = gr.Image(type="pil", label="图片显示")

# 当文件夹输入变化时更新图片列表的函数
def update_image_list(folder_path):
    if os.path.isdir(folder_path):
        images = load_images(folder_path)
        image_dropdown.update(choices=images)
        return images
    else:
        return "该路径不是一个文件夹"

# 当用户从下拉列表中选择一张图片时显示图片的函数
def update_image_output(folder_path, image_name):
    if image_name is not None:
        return show_image(folder_path, image_name)
    else:
        return None

# 创建一个界面
demo = gr.Interface(
    fn=update_image_output,  # 主函数
    inputs=[folder_input, image_dropdown],  # 输入组件：文件夹路径输入框、图片列表下拉菜单
    outputs=image_output,  # 输出组件：图片显示
    live=True,  # 实时更新
)

# 当用户改变文件夹输入时更新图片列表
#folder_input.change(fn=update_image_list, inputs=[folder_input], outputs=[image_dropdown])
demo.launch(share=False, server_name="0.0.0.0", server_port=7070)