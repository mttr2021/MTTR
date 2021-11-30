"""
This script converts the ground-truth annotations of the a2d-sentences dataset to COCO format (for mAP calculation).
This results in a ground-truth JSON file which can be loaded using the pycocotools API.
Note that during evaluation model predictions need to be converted to COCO format as well (check out trainer.py).
"""

import numpy as np
import h5py
import pandas
from os import path
from glob import glob
import json
from tqdm import tqdm
from pycocotools.mask import encode, area
from datasets.a2d_sentences import a2d_sentences_dataset

subset_type = 'test'
dataset_path = './a2d_sentences'
output_path = f'./datasets/a2d_sentences/a2d_sentences_{subset_type}_annotations_in_coco_format.json'


def get_text_annotations(root_path, subset):
    # without 'header == None' pandas will ignore the first sample...
    a2d_data_info = pandas.read_csv(path.join(root_path, 'Release/videoset.csv'), header=None)
    assert len(a2d_data_info) == 3782, f'error: a2d videoset.csv file is missing one or more samples'
    # 'vid', 'label', 'start_time', 'end_time', 'height', 'width', 'total_frames', 'annotated_frames', 'subset'
    a2d_data_info.columns = ['vid', '', '', '', '', '', '', '', 'subset']
    with open(path.join(root_path, 'text_annotations/a2d_missed_videos.txt'), 'r') as f:
        unused_videos = f.read().splitlines()
    subsets = {'train': 0, 'test': 1}
    # filter unused videos and videos which do not belong to our train/test subset:
    used_videos = a2d_data_info[~a2d_data_info.vid.isin(unused_videos) & (a2d_data_info.subset == subsets[subset])]
    used_videos_ids = list(used_videos['vid'])
    text_annotations = pandas.read_csv(path.join(root_path, 'text_annotations/a2d_annotation.txt'))
    # filter the text annotations based on the used videos:
    used_text_annotations = text_annotations[text_annotations.video_id.isin(used_videos_ids)]
    # convert data-frame to list of tuples:
    used_text_annotations = list(used_text_annotations.to_records(index=False))
    return used_text_annotations


def create_a2d_sentences_ground_truth_test_annotations():
    mask_annotations_dir = path.join(dataset_path, 'text_annotations/a2d_annotation_with_instances')
    text_annotations = get_text_annotations(dataset_path, subset_type)

    # Note - it is very important to start counting the instance and category ids from 1 (not 0). This is implicitly
    # expected by pycocotools as it is the convention of the original coco dataset annotations.

    categories_dict = [{'id': 1, 'name': 'dummy_class'}]  # dummy class, as categories are not used/predicted in RVOS

    images_dict = []
    annotations_dict = []
    images_set = set()
    instance_id_counter = 1
    for annot in tqdm(text_annotations):
        video_id, instance_id, text_query = annot
        annot_paths = sorted(glob(path.join(mask_annotations_dir, video_id, '*.h5')))
        for p in annot_paths:
            f = h5py.File(p)
            instances = list(f['instance'])
            try:
                instance_idx = instances.index(int(instance_id))
            # in case this instance does not appear in this frame it has no ground-truth mask, and thus this
            # frame-instance pair is ignored in evaluation, same as SOTA method: CMPC-V. check out:
            # https://github.com/spyflying/CMPC-Refseg/blob/094639b8bf00cc169ea7b49cdf9c87fdfc70d963/CMPC_video/build_A2D_batches.py#L98
            except ValueError:
                continue  # instance_id does not appear in current frame
            mask = f['reMask'][instance_idx] if len(instances) > 1 else np.array(f['reMask'])
            mask = mask.transpose()

            frame_idx = int(p.split('/')[-1].split('.')[0])
            image_id = a2d_sentences_dataset.get_image_id(video_id, frame_idx, instance_id)
            assert image_id not in images_set, f'error: image id: {image_id} appeared twice'
            images_set.add(image_id)
            images_dict.append({'id': image_id, 'height': mask.shape[0], 'width': mask.shape[1]})

            mask_rle = encode(mask)
            mask_rle['counts'] = mask_rle['counts'].decode('ascii')
            mask_area = float(area(mask_rle))
            bbox = f['reBBox'][:, instance_idx] if len(instances) > 1 else np.array(f['reBBox']).squeeze()  # x1y1x2y2 form
            bbox_xywh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            instance_annot = {'id': instance_id_counter,
                              'image_id': image_id,
                              'category_id': 1,  # dummy class, as categories are not used/predicted in ref-vos
                              'segmentation': mask_rle,
                              'area': mask_area,
                              'bbox': bbox_xywh,
                              'iscrowd': 0,
                              }
            annotations_dict.append(instance_annot)
            instance_id_counter += 1
    dataset_dict = {'categories': categories_dict, 'images': images_dict, 'annotations': annotations_dict}
    with open(output_path, 'w') as f:
        json.dump(dataset_dict, f)


if __name__ == '__main__':
    create_a2d_sentences_ground_truth_test_annotations()
