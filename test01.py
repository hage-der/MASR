# -*- coding: gbk -*-
import torch
# pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())


def read_and_save_log(input_file_path, output_file_path):
    """
    ��ȡ��־�ļ��������ݱ��浽�µ��ı��ļ���

    ����:
        input_file_path: ������־�ļ���·��
        output_file_path: ����ı��ļ���·��
    """
    try:
        # �������ļ����ж�ȡ��ʹ�� UTF-8 ����
        with open(input_file_path, 'r', encoding='utf-8') as input_file:
            # ��ȡ��������
            log_content = input_file.read()

            # ������ļ�����д��
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                # ����־����д������ļ�
                output_file.write(log_content)

        print(f"�ɹ���ȡ�ļ� {input_file_path} �������ݱ��浽 {output_file_path}")
        return True

    except FileNotFoundError:
        print(f"�����Ҳ����ļ� {input_file_path}")
        return False
    except PermissionError:
        print(f"����û��Ȩ�޷����ļ� {input_file_path} �� {output_file_path}")
        return False
    except UnicodeDecodeError:
        # ��� UTF-8 ����ʧ�ܣ�����ʹ����������
        try:
            with open(input_file_path, 'r', encoding='gbk') as input_file:
                log_content = input_file.read()

                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(log_content)

            print(f"�ɹ�ʹ�� GBK �����ȡ�ļ� {input_file_path} �������ݱ��浽 {output_file_path}")
            return True
        except Exception as e:
            print(f"����ʹ�� GBK ����ʱ����{str(e)}")
            return False
    except Exception as e:
        print(f"�����ļ�ʱ��������{str(e)}")
        return False


# ʹ��ʾ��
if __name__ == "__main__":
    # �滻Ϊ��ʵ�ʵ���־�ļ�·����Ŀ������ļ�·��
    log_file = "F:/pythoncode/MASR/log/111.log"
    output_file = "F:/pythoncode/MASR/log/111.txt"

    read_and_save_log(log_file, output_file)