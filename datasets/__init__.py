from datasets.a2d_sentences.a2d_sentences_dataset import A2DSentencesDataset
from datasets.jhmdb_sentences.jhmdb_sentences_dataset import JHMDBSentencesDataset
from datasets.refer_youtube_vos.refer_youtube_vos_dataset import ReferYouTubeVOSDataset


def build_dataset(image_set, dataset_name, **kwargs):
    if dataset_name == 'a2d_sentences':
        return A2DSentencesDataset(image_set, **kwargs)
    elif dataset_name == 'jhmdb_sentences':
        return JHMDBSentencesDataset(image_set, **kwargs)
    elif dataset_name == 'ref_youtube_vos':
        return ReferYouTubeVOSDataset(image_set, **kwargs)
    raise ValueError(f'dataset {dataset_name} not supported')
