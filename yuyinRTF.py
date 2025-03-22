# Author : ZY 
# Time : 2025/3/22 21:13 
# ���� :
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
add_arg('configs', str, 'configs/conformer.yml', "�����ļ�")
add_arg('wav_path', str, 'dataset/test.wav', "Ԥ����Ƶ��·��")
add_arg('is_long_audio', bool, False, "�Ƿ�Ϊ������")
add_arg('real_time_demo', bool, False, "�Ƿ�ʹ��ʵʱ����ʶ����ʾ")
add_arg('use_gpu', bool, True, "�Ƿ�ʹ��GPUԤ��")
add_arg('use_pun', bool, False, "�Ƿ��ʶ�����ӱ�����")
add_arg('is_itn', bool, False, "�Ƿ���ı����з���׼��")
add_arg('model_path', str, 'models/conformer_streaming_fbank/inference.pt', "������Ԥ��ģ���ļ�·��")
add_arg('pun_model_dir', str, 'models/pun_models/', "�ӱ����ŵ�ģ���ļ���·��")
add_arg('batch_rtf', bool, False, "�Ƿ���������ݼ�����RTF")
add_arg('dataset_path', str, 'dataset/test/', "�������ݼ�·����������������RTF")
add_arg('warmup_runs', int, 5, "Ԥ�����д���")
args = parser.parse_args()
print_arguments(args=args)

# ��ȡʶ����
predictor = MASRPredictor(configs=args.configs,
                          model_path=args.model_path,
                          use_gpu=args.use_gpu,
                          use_pun=args.use_pun,
                          pun_model_dir=args.pun_model_dir)


# ��ȡ��Ƶʱ��(��)
def get_audio_duration(audio_path):
    """��ȡ��Ƶ�ļ�ʱ��"""
    try:
        with wave.open(audio_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            return duration
    except:
        # �������WAV�ļ�������ʹ��librosa
        try:
            duration = librosa.get_duration(filename=audio_path)
            return duration
        except Exception as e:
            print(f"�޷���ȡ��Ƶʱ��: {e}")
            return 0


# ������ʶ��
def predict_audio():
    # ��ȡ��Ƶʱ��
    audio_duration = get_audio_duration(args.wav_path)
    if audio_duration == 0:
        print("�޷���ȡ��Ƶʱ�����޷�����RTF")
        return

    # Ԥ������(���ʹ��GPU)
    if args.use_gpu and args.warmup_runs > 0:
        print(f"ִ��{args.warmup_runs}��Ԥ������...")
        for _ in range(args.warmup_runs):
            predictor.predict(audio_data=args.wav_path, use_pun=args.use_pun, is_itn=args.is_itn)

    # ��ʼ��ʽ����
    start = time.time()
    result = predictor.predict(audio_data=args.wav_path, use_pun=args.use_pun, is_itn=args.is_itn)
    process_time = time.time() - start
    score, text = result['score'], result['text']

    # ����RTF
    rtf = process_time / audio_duration

    print(f"��Ƶʱ��: {audio_duration:.3f}��")
    print(f"����ʱ��: {process_time:.3f}��")
    print(f"RTF: {rtf:.4f}")
    print(f"ʶ����: {text}")
    print(f"�÷�: {int(score)}")

    return {
        'audio_duration': audio_duration,
        'process_time': process_time,
        'rtf': rtf,
        'text': text,
        'score': score
    }


# ������ʶ��
def predict_long_audio():
    # ��ȡ��Ƶʱ��
    audio_duration = get_audio_duration(args.wav_path)
    if audio_duration == 0:
        print("�޷���ȡ��Ƶʱ�����޷�����RTF")
        return

    # Ԥ������(���ʹ��GPU)
    if args.use_gpu and args.warmup_runs > 0:
        print(f"ִ��{args.warmup_runs}��Ԥ������...")
        for _ in range(args.warmup_runs):
            predictor.predict_long(audio_data=args.wav_path, use_pun=args.use_pun, is_itn=args.is_itn)

    # ��ʼ��ʽ����
    start = time.time()
    result = predictor.predict_long(audio_data=args.wav_path, use_pun=args.use_pun, is_itn=args.is_itn)
    process_time = time.time() - start
    score, text = result['score'], result['text']

    # ����RTF
    rtf = process_time / audio_duration

    print(f"������ʶ����:")
    print(f"��Ƶʱ��: {audio_duration:.3f}��")
    print(f"����ʱ��: {process_time:.3f}��")
    print(f"RTF: {rtf:.4f}")
    print(f"ʶ����: {text}")
    print(f"�÷�: {int(score)}")

    return {
        'audio_duration': audio_duration,
        'process_time': process_time,
        'rtf': rtf,
        'text': text,
        'score': score
    }


# ʵʱʶ��ģ��
def real_time_predict_demo():
    # ��ȡ��Ƶʱ��
    audio_duration = get_audio_duration(args.wav_path)
    if audio_duration == 0:
        print("�޷���ȡ��Ƶʱ�����޷���������RTF")

    # ʶ����ʱ��
    interval_time = 0.5
    CHUNK = int(16000 * interval_time)

    # ��ȡ����
    wf = wave.open(args.wav_path, 'rb')
    channels = wf.getnchannels()
    samp_width = wf.getsampwidth()
    sample_rate = wf.getframerate()

    # ��ȡ��֡������ʱ��
    total_frames = wf.getnframes()
    total_duration = total_frames / sample_rate

    # ����������ʼ��ʱ
    predictor.reset_stream()
    total_process_time = 0

    # Ԥ������
    if args.use_gpu and args.warmup_runs > 0:
        print(f"ִ��{args.warmup_runs}��Ԥ������...")
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

    # ��ʽ����
    data = wf.readframes(CHUNK)
    chunk_count = 0
    final_text = ""

    # ����
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

        # ���㵱ǰ���RTF
        chunk_duration = len(data) / (sample_rate * samp_width * channels)
        chunk_rtf = chunk_process_time / chunk_duration

        data = d
        if result is None:
            continue

        score, text = result['score'], result['text']
        final_text = text  # ��������ʶ����

        print(f"���� {chunk_count}����ʱ����{chunk_duration:.3f}��, ����ʱ�䣺{chunk_process_time:.3f}��, RTF: {chunk_rtf:.4f}")
        print(f"ʶ����: {text}, �÷�: {int(score)}")

    # ��������RTF
    overall_rtf = total_process_time / total_duration

    print("\n��������ʽʶ������")
    print(f"����Ƶʱ��: {total_duration:.3f}��")
    print(f"�ܴ���ʱ��: {total_process_time:.3f}��")
    print(f"����RTF: {overall_rtf:.4f}")
    print(f"����ʶ����: {final_text}")

    # ������ʽʶ��
    predictor.reset_stream()

    return {
        'audio_duration': total_duration,
        'process_time': total_process_time,
        'rtf': overall_rtf,
        'text': final_text
    }


# ��������RTF
def batch_rtf_test():
    """���������ݼ�����RTF����"""
    print(f"�����ݼ� {args.dataset_path} �Ͻ�������RTF����...")

    # ���Ŀ¼�Ƿ����
    if not os.path.exists(args.dataset_path):
        print(f"����: ���ݼ�·�� {args.dataset_path} ������!")
        return

    # ��ȡ������Ƶ�ļ�
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    audio_files = []

    for root, _, files in os.walk(args.dataset_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(root, file))

    if not audio_files:
        print(f"�� {args.dataset_path} ��δ�ҵ���Ƶ�ļ�")
        return

    print(f"�ҵ� {len(audio_files)} ����Ƶ�ļ�")

    # Ԥ������
    if args.use_gpu and args.warmup_runs > 0 and audio_files:
        print(f"ִ��{args.warmup_runs}��Ԥ������...")
        for _ in range(args.warmup_runs):
            predictor.predict(audio_data=audio_files[0], use_pun=args.use_pun, is_itn=args.is_itn)

    # ��ʼ����
    results = []
    total_audio_duration = 0
    total_process_time = 0

    for audio_file in tqdm(audio_files, desc="�������"):
        # ��ȡ��Ƶʱ��
        audio_duration = get_audio_duration(audio_file)
        if audio_duration == 0:
            print(f"���� {audio_file} - �޷���ȡ��Ƶʱ��")
            continue

        # ������Ƶ
        start = time.time()
        result = predictor.predict(audio_data=audio_file, use_pun=args.use_pun, is_itn=args.is_itn)
        process_time = time.time() - start

        # ����RTF
        rtf = process_time / audio_duration

        # ��¼���
        item_result = {
            'file': os.path.basename(audio_file),
            'audio_duration': audio_duration,
            'process_time': process_time,
            'rtf': rtf,
            'text': result['text'],
            'score': result['score']
        }
        results.append(item_result)

        # �ۼ���ʱ���ʹ���ʱ��
        total_audio_duration += audio_duration
        total_process_time += process_time

    # ����ͳ������
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

    # ��ӡͳ�ƽ��
    print("\n====== RTF����ͳ�ƽ�� ======")
    print(f"�����ļ���: {stats['num_files']}")
    print(f"����Ƶʱ��: {stats['total_audio_duration']:.3f}��")
    print(f"�ܴ���ʱ��: {stats['total_process_time']:.3f}��")
    print(f"����RTF: {stats['overall_rtf']:.4f}")
    print(f"ƽ��RTF: {stats['mean_rtf']:.4f}")
    print(f"��λ��RTF: {stats['median_rtf']:.4f}")
    print(f"��СRTF: {stats['min_rtf']:.4f}")
    print(f"���RTF: {stats['max_rtf']:.4f}")
    print(f"RTF��׼��: {stats['std_rtf']:.4f}")
    print(f"ʵʱ����ٷֱ�: {stats['realtime_percentage']:.2f}%")

    # ��������CSV�ļ�
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        output_file = os.path.join(os.path.dirname(args.dataset_path), "rtf_results.csv")
        df.to_csv(output_file, index=False)
        print(f"��ϸ����ѱ�����: {output_file}")

        # ����ͳ�ƽ��ժҪ
        stats_df = pd.DataFrame([stats])
        stats_file = os.path.join(os.path.dirname(args.dataset_path), "rtf_stats.csv")
        stats_df.to_csv(stats_file, index=False)
        print(f"ͳ�ƽ���ѱ�����: {stats_file}")
    except ImportError:
        print("����: δ��װpandas���޷���������CSV�ļ�")

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