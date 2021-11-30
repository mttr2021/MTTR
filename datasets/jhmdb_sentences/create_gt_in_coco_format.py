"""
This script converts the ground-truth annotations of the jhmdb-sentences dataset to COCO format.
This results in a ground-truth JSON file which can be loaded using the pycocotools API.
Note that during evaluation model predictions need to be converted to COCO format as well (check out trainer.py).
"""

import json
import scipy.io
from tqdm import tqdm
from pycocotools.mask import encode, area


def get_image_id(video_id, frame_idx):
    image_id = f'v_{video_id}_f_{frame_idx}'
    return image_id


def create_jhmdb_sentences_ground_truth_annotations(samples_metadata, dataset_coco_gt_format_path, **kwargs):
    # Note - it is very important to start counting the instance and category ids from 1 (not 0). This is implicitly
    # expected by pycocotools as it is the convention of the original coco dataset annotations.
    categories_dict = [{'id': 1, 'name': 'dummy_class'}]  # dummy class, as categories are not used/predicted in RVOS
    images_dict = []
    annotations_dict = []
    images_set = set()
    instance_id_counter = 1
    for sample_metadata in tqdm(samples_metadata):
        video_id, chosen_frame_path, video_masks_path, _, text_query = sample_metadata

        chosen_frame_idx = int(chosen_frame_path.split('/')[-1].split('.')[0])
        # read the instance masks:
        all_video_masks = scipy.io.loadmat(video_masks_path)['part_mask'].transpose(2, 0, 1)
        # note that to take the center-frame corresponding mask we switch to 0-indexing:
        mask = all_video_masks[chosen_frame_idx - 1]

        image_id = get_image_id(video_id, chosen_frame_idx)
        assert image_id not in images_set, f'error: image id: {image_id} appeared twice'
        images_set.add(image_id)
        images_dict.append({'id': image_id, 'height': mask.shape[0], 'width': mask.shape[1]})

        mask_rle = encode(mask)
        mask_rle['counts'] = mask_rle['counts'].decode('ascii')
        mask_area = float(area(mask_rle))
        instance_annot = {'id': instance_id_counter,
                          'image_id': image_id,
                          'category_id': 1,  # dummy class, as categories are not used/predicted in RVOS
                          'segmentation': mask_rle,
                          'area': mask_area,
                          'iscrowd': 0,
                          }
        annotations_dict.append(instance_annot)
        instance_id_counter += 1
    dataset_dict = {'categories': categories_dict, 'images': images_dict, 'annotations': annotations_dict}
    with open(dataset_coco_gt_format_path, 'w') as f:
        json.dump(dataset_dict, f)
