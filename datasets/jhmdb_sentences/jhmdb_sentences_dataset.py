import torch
from torch.utils.data import Dataset
import torch.distributed as dist
import pandas
from os import path
from glob import glob
from tqdm import tqdm
import random
import scipy.io
from PIL import Image
import json
from misc import nested_tensor_from_videos_list
from datasets.a2d_sentences.a2d_sentences_dataset import A2dSentencesTransforms
from datasets.jhmdb_sentences.create_gt_in_coco_format import create_jhmdb_sentences_ground_truth_annotations, get_image_id


class JHMDBSentencesDataset(Dataset):
    """
    A Torch dataset for JHMDB-Sentences.
    For more information check out: https://kgavrilyuk.github.io/publication/actor_action/ or the original paper at:
    https://arxiv.org/abs/1803.07485
    """
    def __init__(self, subset_type: str = 'test', dataset_path: str = './jhmdb_sentences', window_size=8,
                 generate_new_samples_metadata=True, distributed=False, **kwargs):
        super(JHMDBSentencesDataset, self).__init__()
        assert subset_type in ['train', 'test'], 'error: unsupported subset type. supported: train (dummy), test'
        if subset_type == 'train':
            self.collator = None
            self.samples_metadata = [0]
            return  # JHMDB-Sentences is only for evaluation (test), not training, so the training dataset is a dummy
        self.subset_type = subset_type
        self.samples_metadata = self.get_samples_metadata(dataset_path, generate_new_samples_metadata, distributed)
        self.window_size = window_size
        self.transforms = A2dSentencesTransforms(subset_type, **kwargs)  # same transformations as used in a2d-sentences
        self.collator = Collator()
        # create ground-truth annotations for the evaluation process:
        if subset_type == 'test':
            if (distributed and dist.get_rank() == 0) or not distributed:
                create_jhmdb_sentences_ground_truth_annotations(self.samples_metadata, **kwargs)
            if distributed:
                dist.barrier()

    @staticmethod
    def get_samples_metadata(root_path, generate_new_samples_metadata, distributed):
        samples_metadata_file_path = f'./datasets/jhmdb_sentences/jhmdb_sentences_samples_metadata.json'
        if not generate_new_samples_metadata:  # load existing metadata file
            with open(samples_metadata_file_path, 'r') as f:
                samples_metadata = [tuple(a) for a in json.load(f)]
                return samples_metadata
        if (distributed and dist.get_rank() == 0) or not distributed:
            print(f'creating jhmdb-sentences samples metadata...')
            text_annotations = pandas.read_csv(path.join(root_path, 'jhmdb_annotation.txt'))
            assert len(text_annotations) == 928, 'error: jhmdb_annotation.txt is missing one or more samples.'
            text_annotations = list(text_annotations.to_records(index=False))
            used_videos_ids = set([vid_id for vid_id, _ in text_annotations])
            video_frames_folder_paths = sorted(glob(path.join(root_path, 'Rename_Images', '*', '*')))
            video_frames_folder_paths = {p.split('/')[-1]: p for p in video_frames_folder_paths if p.split('/')[-1] in used_videos_ids}
            video_masks_folder_paths = sorted(glob(path.join(root_path, 'puppet_mask', '*', '*', 'puppet_mask.mat')))
            video_masks_folder_paths = {p.split('/')[-2]: p for p in video_masks_folder_paths if p.split('/')[-2] in used_videos_ids}
            samples_metadata = []
            for video_id, text_query in tqdm(text_annotations):
                video_frames_paths = sorted(glob(path.join(video_frames_folder_paths[video_id], '*.png')))
                video_masks_path = video_masks_folder_paths[video_id]
                video_total_masks = len(scipy.io.loadmat(video_masks_path)['part_mask'].transpose(2, 0, 1))
                # some of the last frames in the video may not have masks and thus cannot be used for evaluation
                # so we ignore them:
                video_frames_paths = video_frames_paths[:video_total_masks]
                video_total_frames = len(video_frames_paths)
                chosen_frames_paths = sorted(random.sample(video_frames_paths, 3))  # sample 3 frames randomly
                for frame_path in chosen_frames_paths:
                    samples_metadata.append((video_id, frame_path, video_masks_path, video_total_frames, text_query))
            with open(samples_metadata_file_path, 'w') as f:
                json.dump(samples_metadata, f)
        # in the distributed setting the metadata is created on the main process and then synced between all the
        # processes. this is necessary as otherwise different processes may randomly sample different frames from the
        # dataset.
        if distributed:
            dist.barrier()
            with open(samples_metadata_file_path, 'r') as f:
                samples_metadata = [tuple(a) for a in json.load(f)]
        return samples_metadata

    def __getitem__(self, idx):
        video_id, chosen_frame_path, video_masks_path, video_total_frames, text_query = self.samples_metadata[idx]
        text_query = " ".join(text_query.lower().split())  # clean up the text query

        # read the source window frames:
        chosen_frame_idx = int(chosen_frame_path.split('/')[-1].split('.')[0])
        # get a window of window_size frames with frame chosen_frame_idx in the middle.
        start_idx, end_idx = chosen_frame_idx - self.window_size // 2, chosen_frame_idx + (self.window_size + 1) // 2
        frame_indices = list(range(start_idx, end_idx))  # note that jhmdb-sentences frames are 1-indexed
        # extract the window source frames:
        source_frames = []
        for i in frame_indices:
            i = min(max(i, 1), video_total_frames)  # pad out of range indices with edge frames
            p = '/'.join(chosen_frame_path.split('/')[:-1]) + f'/{i:05d}.png'
            source_frames.append(Image.open(p).convert('RGB'))

        # read the instance masks:
        all_video_masks = scipy.io.loadmat(video_masks_path)['part_mask'].transpose(2, 0, 1)
        # note that to take the center-frame corresponding mask we switch to 0-indexing:
        instance_mask = torch.tensor(all_video_masks[chosen_frame_idx - 1]).unsqueeze(0)

        # create the target dict for the center frame:
        target = {'masks': instance_mask,
                  'orig_size': instance_mask.shape[-2:],  # original frame shape without any augmentations
                  # size with augmentations, will be changed inside transforms if necessary
                  'size': instance_mask.shape[-2:],
                  'referred_instance_idx': torch.zeros(1, dtype=torch.long),  # idx in 'masks' of the text referred instance
                  'iscrowd': torch.zeros(1),  # for compatibility with DETR COCO transforms
                  'image_id': get_image_id(video_id, chosen_frame_idx)}

        # create dummy targets for adjacent frames:
        targets = self.window_size * [None]
        center_frame_idx = self.window_size // 2
        targets[center_frame_idx] = target

        source_frames, targets, text_query = self.transforms(source_frames, targets, text_query)
        return source_frames, targets, text_query

    def __len__(self):
        return len(self.samples_metadata)


class Collator:
    def __call__(self, batch):
        samples, targets, text_queries = list(zip(*batch))
        samples = nested_tensor_from_videos_list(samples)  # [T, B, C, H, W]
        # convert targets to a list of tuples. outer list - time steps, inner tuples - time step batch
        targets = list(zip(*targets))
        batch_dict = {
            'samples': samples,
            'targets': targets,
            'text_queries': text_queries
        }
        return batch_dict
