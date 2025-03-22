# Author : ZY 
# Time : 2025/3/22 21:13 
# 内容 :
import argparse
import functools
import time
import wave
import os
import numpy as np
import librosa
from tqdm import tqdm

from masr.predict import MASRPredictor
from masr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs', str, 'configs/conformer.yml', "配置文件")
add_arg('wav_path', str, 'dataset/test.wav', "预测音频的路径")
add_arg('is_long_audio', bool, False, "是否为长语音")
add_arg('real_time_demo', bool, False, "是否使用实时语音识别演示")
add_arg('use_gpu', bool, True, "是否使用GPU预测")
add_arg('use_pun', bool, False, "是否给识别结果加标点符号")
add_arg('is_itn', bool, False, "是否对文本进行反标准化")
add_arg('model_path', str, 'models/conformer_streaming_fbank/inference.pt', "导出的预测模型文件路径")
add_arg('pun_model_dir', str, 'models/pun_models/', "加标点符号的模型文件夹路径")
add_arg('batch_rtf', bool, False, "是否对整个数据集计算RTF")
add_arg('dataset_path', str, 'dataset/test/', "测试数据集路径，用于批量计算RTF")
add_arg('warmup_runs', int, 5, "预热运行次数")
args = parser.parse_args()
print_arguments(args=args)

# 获取识别器
predictor = MASRPredictor(configs=args.configs,
                          model_path=args.model_path,
                          use_gpu=args.use_gpu,
                          use_pun=args.use_pun,
                          pun_model_dir=args.pun_model_dir)


# 获取音频时长(秒)
def get_audio_duration(audio_path):
    """获取音频文件时长"""
    try:
        with wave.open(audio_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            return duration
    except:
        # 如果不是WAV文件，尝试使用librosa
        try:
            duration = librosa.get_duration(filename=audio_path)
            return duration
        except Exception as e:
            print(f"无法获取音频时长: {e}")
            return 0


# 短语音识别
def predict_audio():
    # 获取音频时长
    audio_duration = get_audio_duration(args.wav_path)
    if audio_duration == 0:
        print("无法获取音频时长，无法计算RTF")
        return

    # 预热运行(如果使用GPU)
    if args.use_gpu and args.warmup_runs > 0:
        print(f"执行{args.warmup_runs}次预热运行...")
        for _ in range(args.warmup_runs):
            predictor.predict(audio_data=args.wav_path, use_pun=args.use_pun, is_itn=args.is_itn)

    # 开始正式测量
    start = time.time()
    result = predictor.predict(audio_data=args.wav_path, use_pun=args.use_pun, is_itn=args.is_itn)
    process_time = time.time() - start
    score, text = result['score'], result['text']

    # 计算RTF
    rtf = process_time / audio_duration

    print(f"音频时长: {audio_duration:.3f}秒")
    print(f"处理时间: {process_time:.3f}秒")
    print(f"RTF: {rtf:.4f}")
    print(f"识别结果: {text}")
    print(f"得分: {int(score)}")

    return {
        'audio_duration': audio_duration,
        'process_time': process_time,
        'rtf': rtf,
        'text': text,
        'score': score
    }


# 长语音识别
def predict_long_audio():
    # 获取音频时长
    audio_duration = get_audio_duration(args.wav_path)
    if audio_duration == 0:
        print("无法获取音频时长，无法计算RTF")
        return

    # 预热运行(如果使用GPU)
    if args.use_gpu and args.warmup_runs > 0:
        print(f"执行{args.warmup_runs}次预热运行...")
        for _ in range(args.warmup_runs):
            predictor.predict_long(audio_data=args.wav_path, use_pun=args.use_pun, is_itn=args.is_itn)

    # 开始正式测量
    start = time.time()
    result = predictor.predict_long(audio_data=args.wav_path, use_pun=args.use_pun, is_itn=args.is_itn)
    process_time = time.time() - start
    score, text = result['score'], result['text']

    # 计算RTF
    rtf = process_time / audio_duration

    print(f"长语音识别结果:")
    print(f"音频时长: {audio_duration:.3f}秒")
    print(f"处理时间: {process_time:.3f}秒")
    print(f"RTF: {rtf:.4f}")
    print(f"识别结果: {text}")
    print(f"得分: {int(score)}")

    return {
        'audio_duration': audio_duration,
        'process_time': process_time,
        'rtf': rtf,
        'text': text,
        'score': score
    }


# 实时识别模拟
def real_time_predict_demo():
    # 获取音频时长
    audio_duration = get_audio_duration(args.wav_path)
    if audio_duration == 0:
        print("无法获取音频时长，无法计算整体RTF")

    # 识别间隔时间
    interval_time = 0.5
    CHUNK = int(16000 * interval_time)

    # 读取数据
    wf = wave.open(args.wav_path, 'rb')
    channels = wf.getnchannels()
    samp_width = wf.getsampwidth()
    sample_rate = wf.getframerate()

    # 获取总帧数和总时长
    total_frames = wf.getnframes()
    total_duration = total_frames / sample_rate

    # 重置流并开始计时
    predictor.reset_stream()
    total_process_time = 0

    # 预热运行
    if args.use_gpu and args.warmup_runs > 0:
        print(f"执行{args.warmup_runs}次预热运行...")
        temp_wf = wave.open(args.wav_path, 'rb')
        temp_data = temp_wf.readframes(CHUNK)
        for _ in range(args.warmup_runs):
            predictor.predict_stream(audio_data=temp_data,
                                     use_pun=args.use_pun,
                                     is_itn=args.is_itn,
                                     is_end=True,
                                     channels=channels,
                                     samp_width=samp_width,
                                     sample_rate=sample_rate)
            predictor.reset_stream()
        temp_wf.close()

    # 正式测量
    data = wf.readframes(CHUNK)
    chunk_count = 0
    final_text = ""

    # 播放
    while data != b'':
        chunk_count += 1
        start = time.time()
        d = wf.readframes(CHUNK)
        is_end = d == b''

        result = predictor.predict_stream(
            audio_data=data,
            use_pun=args.use_pun,
            is_itn=args.is_itn,
            is_end=is_end,
            channels=channels,
            samp_width=samp_width,
            sample_rate=sample_rate
        )

        chunk_process_time = time.time() - start
        total_process_time += chunk_process_time

        # 计算当前块的RTF
        chunk_duration = len(data) / (sample_rate * samp_width * channels)
        chunk_rtf = chunk_process_time / chunk_duration

        data = d
        if result is None:
            continue

        score, text = result['score'], result['text']
        final_text = text  # 保存最后的识别结果

        print(f"【块 {chunk_count}】：时长：{chunk_duration:.3f}秒, 处理时间：{chunk_process_time:.3f}秒, RTF: {chunk_rtf:.4f}")
        print(f"识别结果: {text}, 得分: {int(score)}")

    # 计算整体RTF
    overall_rtf = total_process_time / total_duration

    print("\n【整体流式识别结果】")
    print(f"总音频时长: {total_duration:.3f}秒")
    print(f"总处理时间: {total_process_time:.3f}秒")
    print(f"整体RTF: {overall_rtf:.4f}")
    print(f"最终识别结果: {final_text}")

    # 重置流式识别
    predictor.reset_stream()

    return {
        'audio_duration': total_duration,
        'process_time': total_process_time,
        'rtf': overall_rtf,
        'text': final_text
    }


# 批量测试RTF
def batch_rtf_test():
    """对整个数据集进行RTF测试"""
    print(f"在数据集 {args.dataset_path} 上进行批量RTF测试...")

    # 检查目录是否存在
    if not os.path.exists(args.dataset_path):
        print(f"错误: 数据集路径 {args.dataset_path} 不存在!")
        return

    # 获取所有音频文件
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    audio_files = []

    for root, _, files in os.walk(args.dataset_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(root, file))

    if not audio_files:
        print(f"在 {args.dataset_path} 中未找到音频文件")
        return

    print(f"找到 {len(audio_files)} 个音频文件")

    # 预热运行
    if args.use_gpu and args.warmup_runs > 0 and audio_files:
        print(f"执行{args.warmup_runs}次预热运行...")
        for _ in range(args.warmup_runs):
            predictor.predict(audio_data=audio_files[0], use_pun=args.use_pun, is_itn=args.is_itn)

    # 开始测试
    results = []
    total_audio_duration = 0
    total_process_time = 0

    for audio_file in tqdm(audio_files, desc="处理进度"):
        # 获取音频时长
        audio_duration = get_audio_duration(audio_file)
        if audio_duration == 0:
            print(f"跳过 {audio_file} - 无法获取音频时长")
            continue

        # 处理音频
        start = time.time()
        result = predictor.predict(audio_data=audio_file, use_pun=args.use_pun, is_itn=args.is_itn)
        process_time = time.time() - start

        # 计算RTF
        rtf = process_time / audio_duration

        # 记录结果
        item_result = {
            'file': os.path.basename(audio_file),
            'audio_duration': audio_duration,
            'process_time': process_time,
            'rtf': rtf,
            'text': result['text'],
            'score': result['score']
        }
        results.append(item_result)

        # 累计总时长和处理时间
        total_audio_duration += audio_duration
        total_process_time += process_time

    # 计算统计数据
    rtf_values = [r['rtf'] for r in results]
    overall_rtf = total_process_time / total_audio_duration

    stats = {
        'mean_rtf': np.mean(rtf_values),
        'median_rtf': np.median(rtf_values),
        'min_rtf': np.min(rtf_values),
        'max_rtf': np.max(rtf_values),
        'std_rtf': np.std(rtf_values),
        'total_audio_duration': total_audio_duration,
        'total_process_time': total_process_time,
        'overall_rtf': overall_rtf,
        'num_files': len(results),
        'realtime_percentage': (np.array(rtf_values) < 1.0).mean() * 100
    }

    # 打印统计结果
    print("\n====== RTF测试统计结果 ======")
    print(f"测试文件数: {stats['num_files']}")
    print(f"总音频时长: {stats['total_audio_duration']:.3f}秒")
    print(f"总处理时间: {stats['total_process_time']:.3f}秒")
    print(f"整体RTF: {stats['overall_rtf']:.4f}")
    print(f"平均RTF: {stats['mean_rtf']:.4f}")
    print(f"中位数RTF: {stats['median_rtf']:.4f}")
    print(f"最小RTF: {stats['min_rtf']:.4f}")
    print(f"最大RTF: {stats['max_rtf']:.4f}")
    print(f"RTF标准差: {stats['std_rtf']:.4f}")
    print(f"实时处理百分比: {stats['realtime_percentage']:.2f}%")

    # 保存结果到CSV文件
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        output_file = os.path.join(os.path.dirname(args.dataset_path), "rtf_results.csv")
        df.to_csv(output_file, index=False)
        print(f"详细结果已保存至: {output_file}")

        # 创建统计结果摘要
        stats_df = pd.DataFrame([stats])
        stats_file = os.path.join(os.path.dirname(args.dataset_path), "rtf_stats.csv")
        stats_df.to_csv(stats_file, index=False)
        print(f"统计结果已保存至: {stats_file}")
    except ImportError:
        print("警告: 未安装pandas，无法保存结果到CSV文件")

    return stats, results


if __name__ == "__main__":
    if args.batch_rtf:
        batch_rtf_test()
    elif args.real_time_demo:
        real_time_predict_demo()
    else:
        if args.is_long_audio:
            predict_long_audio()
        else:
            predict_audio()