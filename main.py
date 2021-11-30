import argparse
import torch
from trainer import Trainer
import ruamel.yaml
import os
import wandb

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '12356'


def run(process_id, args):
    with open(args.config_path) as f:
        config = ruamel.yaml.safe_load(f)
    config = {k: v['value'] for k, v in config.items()}
    config = {**config, **vars(args)}
    config = argparse.Namespace(**config)
    trainer = Trainer(config, process_id, device_id=args.device_ids[process_id], num_processes=args.num_devices)
    if config.running_mode == 'train':
        trainer.train()
    else:  # eval mode:
        model_state_dict = torch.load(config.checkpoint_path)
        if 'model_state_dict' in model_state_dict.keys():
            model_state_dict = model_state_dict['model_state_dict']
        from torch.nn.parallel import DistributedDataParallel as DDP
        model_without_ddp = trainer.model.module if isinstance(trainer.model, DDP) else trainer.model
        model_without_ddp.load_state_dict(model_state_dict, strict=True)
        trainer.evaluate()
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MTTR training and evaluation')
    parser.add_argument('--config_path', '-c', required=True,
                        help='path to configuration file')
    parser.add_argument('--running_mode', '-rm', choices=['train', 'eval'], required=True,
                        help="mode to run, either 'train' or 'eval'")
    parser.add_argument('--window_size', '-ws', type=int,
                        help='window length to use during training/evaluation.'
                             'note - in Refer-YouTube-VOS this parameter is used only during training, as'
                             ' during evaluation full-length videos (all annotated frames) are used.')
    parser.add_argument('--batch_size', '-bs', type=int, required=True,
                        help='training batch size per device')
    parser.add_argument('--eval_batch_size', '-ebs', type=int,
                        help='evaluation batch size per device. '
                             'if not provided training batch size will be used instead.')
    parser.add_argument('--checkpoint_path', '-ckpt', type=str,
                        help='path of checkpoint file to load for evaluation purposes')
    gpu_params_group = parser.add_mutually_exclusive_group(required=True)
    gpu_params_group.add_argument('--num_gpus', '-ng', type=int, default=argparse.SUPPRESS,
                                  help='number of CUDA gpus to run on. mutually exclusive with \'gpu_ids\'')
    gpu_params_group.add_argument('--gpu_ids', '-gids', type=int, nargs='+', default=argparse.SUPPRESS,
                                  help='ids of GPUs to run on. mutually exclusive with \'num_gpus\'')
    gpu_params_group.add_argument('--cpu', '-cpu', action='store_true', default=argparse.SUPPRESS,
                                  help='run on CPU. Not recommended, but could be helpful for debugging if no GPU is'
                                       'available.')
    args = parser.parse_args()

    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size

    if hasattr(args, 'num_gpus'):
        args.num_devices = max(min(args.num_gpus, torch.cuda.device_count()), 1)
        args.device_ids = list(range(args.num_gpus))
    elif hasattr(args, 'gpu_ids'):
        for gpu_id in args.gpu_ids:
            assert 0 <= gpu_id <= torch.cuda.device_count() - 1, \
                f'error: gpu ids must be between 0 and {torch.cuda.device_count() - 1}'
        args.num_devices = len(args.gpu_ids)
        args.device_ids = args.gpu_ids
    else:  # cpu
        args.device_ids = ['cpu']
        args.num_devices = 1

    if args.num_devices > 1:
        torch.multiprocessing.spawn(run, nprocs=args.num_devices, args=(args,))
    else:  # run on a single GPU or CPU
        run(process_id=0, args=args)
