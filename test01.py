# -*- coding: gbk -*-
import torch
# pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())


def read_and_save_log(input_file_path, output_file_path):
    """
    读取日志文件并将内容保存到新的文本文件中

    参数:
        input_file_path: 输入日志文件的路径
        output_file_path: 输出文本文件的路径
    """
    try:
        # 打开输入文件进行读取，使用 UTF-8 编码
        with open(input_file_path, 'r', encoding='utf-8') as input_file:
            # 读取所有内容
            log_content = input_file.read()

            # 打开输出文件进行写入
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                # 将日志内容写入输出文件
                output_file.write(log_content)

        print(f"成功读取文件 {input_file_path} 并将内容保存到 {output_file_path}")
        return True

    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file_path}")
        return False
    except PermissionError:
        print(f"错误：没有权限访问文件 {input_file_path} 或 {output_file_path}")
        return False
    except UnicodeDecodeError:
        # 如果 UTF-8 解码失败，尝试使用其他编码
        try:
            with open(input_file_path, 'r', encoding='gbk') as input_file:
                log_content = input_file.read()

                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(log_content)

            print(f"成功使用 GBK 编码读取文件 {input_file_path} 并将内容保存到 {output_file_path}")
            return True
        except Exception as e:
            print(f"尝试使用 GBK 编码时出错：{str(e)}")
            return False
    except Exception as e:
        print(f"处理文件时发生错误：{str(e)}")
        return False


# 使用示例
if __name__ == "__main__":
    # 替换为你实际的日志文件路径和目标输出文件路径
    log_file = "F:/pythoncode/MASR/log/111.log"
    output_file = "F:/pythoncode/MASR/log/111.txt"

    read_and_save_log(log_file, output_file)