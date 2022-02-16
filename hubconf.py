"""
This is a Pytorch Hub configuration file that is required to loading MTTR in Colab/Hugging-Face-Spaces.
Check out https://pytorch.org/docs/stable/hub.html for more info.
"""
import torch
from torch.hub import get_dir, download_url_to_file
from torchvision.datasets.utils import download_file_from_google_drive
from models import build_model
import argparse
import os
import wget
import base64
import ruamel.yaml

dependencies = ['einops', 'pycocotools', 'ruamel.yaml', 'timm', 'torch', 'transformers.models']

def create_onedrive_directdownload (onedrive_link):
    data_bytes64 = base64.b64encode(bytes(onedrive_link, 'utf-8'))
    data_bytes64_String = data_bytes64.decode('utf-8').replace('/','_').replace('+','-').rstrip("=")
    resultUrl = f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
    return resultUrl


def get_refer_youtube_vos_config(config_dir=None):
    if config_dir is None:
        hub_dir = get_dir()
        config_dir = os.path.join(hub_dir, 'configs')
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, 'refer_youtube_vos.yaml')
    config_url = 'https://raw.githubusercontent.com/mttr2021/MTTR/main/configs/refer_youtube_vos.yaml'
    download_url_to_file(config_url, config_path)
    with open(config_path) as f:
        config = ruamel.yaml.safe_load(f)
    return config


def mttr_refer_youtube_vos(get_weights=True, config=None, config_dir=None, args=None, running_mode='eval'):
    if config is None:
        config = get_refer_youtube_vos_config(config_dir)
    config = {k: v['value'] for k, v in config.items()}
    config['device'] = 'cpu'
    config['running_mode'] = running_mode
    if args is not None:
        config = {**config, **vars(args)}
    config = argparse.Namespace(**config)
    model, _, postprocessor = build_model(config)
    if get_weights:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')
        os.makedirs(model_dir, exist_ok=True)
        checkpoint_path = os.path.join(model_dir, 'refer-youtube-vos_window-12.pth.tar')
        if not os.path.exists(checkpoint_path):
            ckpt_url = 'https://1drv.ms/u/s!AlRIP8CVycWEaZj40fXjNP6d0DU'
            ckpt_direct_url = create_onedrive_directdownload(ckpt_url)
            wget.download(ckpt_direct_url, checkpoint_path)
        model_state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in model_state_dict.keys():
            model_state_dict = model_state_dict['model_state_dict']
        model.load_state_dict(model_state_dict, strict=True)
    return model, postprocessor
