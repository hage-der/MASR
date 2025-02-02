import argparse
import functools
import os

from masr.trainer import MASRTrainer
from masr.utils.utils import add_arguments, print_arguments
# ssh -p 33280 root@connect.bjb1.seetacloud.com
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/conformer.yml',       '配置文件')
add_arg("local_rank",       int,    0,                             '多卡训练的本地GPU')
add_arg("use_gpu",          bool,   True,                          '是否使用GPU训练')
add_arg('augment_conf_path',str,    'configs/augmentation.json',   '数据增强的配置文件，为json格式')
add_arg('save_model_path',  str,    'models/',                  '模型保存的路径')
add_arg('resume_model',     str,    None,                       '恢复训练，当为None则不使用预训练模型，否则指定模型路径')
add_arg('pretrained_model', str,    None,                       '预训练模型的路径，当为None则不使用预训练模型')
args = parser.parse_args()

if int(os.environ.get('LOCAL_RANK', 0)) == 0:
    print_arguments(args=args)

# 获取训练器
trainer = MASRTrainer(configs=args.configs, use_gpu=args.use_gpu)
# Train epoch: [1/100], batch: [100/4513], loss: 18.09715, learning_rate: 0.00000080, reader_cost: 0.1145, batch_cost: 0.1469, ips: 71.2360 speech/sec, eta: 1 day, 4:09:01
trainer.train(save_model_path=args.save_model_path,
              resume_model=args.resume_model,
              pretrained_model=args.pretrained_model,
              augment_conf_path=args.augment_conf_path)
