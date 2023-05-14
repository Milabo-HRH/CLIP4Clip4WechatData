from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np

import copy
import json
import random
import zipfile
import torch
from io import BytesIO
from category_id_map import category_id_to_lv1id, category_id_to_lv2id, CATEGORY_ID_TO_FIRID, CATEGORY_ID_TO_SECID

class Wechat_DataLoader(Dataset):
    def __init__(
            self,
            json_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=32,
            ocr=False,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            subset="train",
            num_thread_reader=1,
    ):
        # todo1: need to preprocess the data
        # self.csv = pd.read_csv(csv_path)
        if subset == "train":   
            self.json_path = os.path.join(json_path, "pretrain.json")
        else:
            self.json_path = os.path.join(json_path, "labeled.json")
        self.data = json.load(open(self.json_path, 'r'))
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.zip_feat_path = features_path
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]
        self.num_workers = num_thread_reader
        if self.num_workers > 0:
            # lazy initialization for zip_handler to avoid multiprocessing-reading error
            self.handles = [None for _ in range(self.num_workers)]
        else:
            self.handles = zipfile.ZipFile(self.zip_feat_path, 'r')
        self.sample_len = len(self.data)
        self.ocr = ocr
        # print(self.data[0])
        # self.data = {key: [d.get(key, None) for d in self.data] for key in self.data[0]}

        # train_video_ids = self.data['id']
        
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "[CLS]", "SEP_TOKEN": "[SEP]",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)
        vid = self.data[video_id]['id']
        for i, video_id in enumerate(choice_video_ids):
            if self.ocr:
                #todo2: complete the ocr part
                words = self.tokenizer.tokenize("")
                pass
            else:
                words = self.tokenizer.tokenize(self.data[video_id]['title'])

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, vid



    def _get_rawvideo(self, vid):
        if self.num_workers > 0:
            worker_id = torch.utils.data.get_worker_info().id
            if self.handles[worker_id] is None:
                self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, 'r')
            handle = self.handles[worker_id]
        else:
            handle = self.handles
        raw_feats = np.load(BytesIO(handle.read(name=f'{vid}.npy')), allow_pickle=True)
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape

        feat = np.zeros((self.max_frames, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frames,), dtype=np.int32)
        if num_frames <= self.max_frames:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            # if self.test_mode:
                # uniformly sample when test mode is True
            step = num_frames // self.max_frames
            select_inds = list(range(0, num_frames, step))
            select_inds = select_inds[:self.max_frames]
        
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        # mask = torch.LongTensor(mask)
        return feat, mask
        

    def __getitem__(self, idx):
        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(idx)
        video, video_mask = self._get_rawvideo(choice_video_ids)
        return pairs_text, pairs_mask, pairs_segment, video, video_mask

class Wechat_Dataloader_finetune(Dataset):
    def __init__(
            self,
            json_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=32,
            ocr=False,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            subset="val",
            num_thread_reader=1,
    ):
        self.json_path = os.path.join(json_path, "train.json")
        self.data = json.load(open(self.json_path, 'r'))
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.zip_feat_path = features_path
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]
        self.num_workers = num_thread_reader
        if self.num_workers > 0:
            # lazy initialization for zip_handler to avoid multiprocessing-reading error
            self.handles = [None for _ in range(self.num_workers)]
        else:
            self.handles = zipfile.ZipFile(self.zip_feat_path, 'r')
        self.sample_len = len(self.data)
        self.ocr = ocr
        # print(self.data[0])
        # self.data = {key: [d.get(key, None) for d in self.data] for key in self.data[0]}

        # train_video_ids = self.data['id']
        
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "[CLS]", "SEP_TOKEN": "[SEP]",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id):
        choice_video_ids = [video_id]
        k = len(choice_video_ids)
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)
        # vid = self.data[video_id]['id']
        for i, video_id in enumerate(choice_video_ids):
            if self.ocr:
                #todo2: complete the ocr part
                words = self.tokenizer.tokenize("")
                pass
            else:
                words = self.tokenizer.tokenize(self.data[video_id]['title'])

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment



    def _get_rawvideo(self, video_id):
        vid = vid = self.data[video_id]['id']
        if self.num_workers > 0:
            worker_id = torch.utils.data.get_worker_info().id
            if self.handles[worker_id] is None:
                self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, 'r')
            handle = self.handles[worker_id]
        else:
            handle = self.handles
        raw_feats = np.load(BytesIO(handle.read(name=f'{vid}.npy')), allow_pickle=True)
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape

        feat = np.zeros((self.max_frames, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frames,), dtype=np.int32)
        if num_frames <= self.max_frames:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            # if self.test_mode:
                # uniformly sample when test mode is True
            step = num_frames // self.max_frames
            select_inds = list(range(0, num_frames, step))
            select_inds = select_inds[:self.max_frames]
        
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat, mask
        

    def __getitem__(self, idx):
        pairs_text, pairs_mask, pairs_segment = self._get_text(idx)
        video, video_mask = self._get_rawvideo(idx)
        labels = {}
        label = category_id_to_lv2id(self.data[idx]['category_id'])
        labels['label'] = torch.LongTensor([label])
        labels['label_v1'] = torch.LongTensor([CATEGORY_ID_TO_FIRID[self.data[idx]['category_id'][:2]]])
        labels['label_v2'] = torch.LongTensor([CATEGORY_ID_TO_SECID[self.data[idx]['category_id']]])
        return pairs_text, pairs_mask, pairs_segment, video, video_mask, labels
  
class Wechat_Dataloader_eval(Dataset):
    def __init__(
            self,
            json_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=32,
            ocr=False,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            subset="val",
            num_thread_reader=1,
    ):
        self.json_path = os.path.join(json_path, "test.json")
        self.data = json.load(open(self.json_path, 'r'))
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.zip_feat_path = features_path
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]
        self.num_workers = num_thread_reader
        if self.num_workers > 0:
            # lazy initialization for zip_handler to avoid multiprocessing-reading error
            self.handles = [None for _ in range(self.num_workers)]
        else:
            self.handles = zipfile.ZipFile(self.zip_feat_path, 'r')
        self.sample_len = len(self.data)
        self.ocr = ocr
        # print(self.data[0])
        # self.data = {key: [d.get(key, None) for d in self.data] for key in self.data[0]}

        # train_video_ids = self.data['id']
        
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "[CLS]", "SEP_TOKEN": "[SEP]",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id):
        choice_video_ids = [video_id]
        k = len(choice_video_ids)
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)
        # vid = self.data[video_id]['id']
        for i, video_id in enumerate(choice_video_ids):
            if self.ocr:
                #todo2: complete the ocr part
                words = self.tokenizer.tokenize("")
                pass
            else:
                words = self.tokenizer.tokenize(self.data[video_id]['title'])

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment



    def _get_rawvideo(self, video_id):
        vid = vid = self.data[video_id]['id']
        if self.num_workers > 0:
            worker_id = torch.utils.data.get_worker_info().id
            if self.handles[worker_id] is None:
                self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, 'r')
            handle = self.handles[worker_id]
        else:
            handle = self.handles
        raw_feats = np.load(BytesIO(handle.read(name=f'{vid}.npy')), allow_pickle=True)
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape

        feat = np.zeros((self.max_frames, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frames,), dtype=np.int32)
        if num_frames <= self.max_frames:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            # if self.test_mode:
                # uniformly sample when test mode is True
            step = num_frames // self.max_frames
            select_inds = list(range(0, num_frames, step))
            select_inds = select_inds[:self.max_frames]
        
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat, mask
        

    def __getitem__(self, idx):
        pairs_text, pairs_mask, pairs_segment = self._get_text(idx)
        video, video_mask = self._get_rawvideo(idx)
        labels = {}
        label = category_id_to_lv2id(self.data[idx]['category_id'])
        labels['label'] = torch.LongTensor([label])
        labels['label_v1'] = torch.LongTensor([CATEGORY_ID_TO_FIRID[self.data[idx]['category_id'][:2]]])
        labels['label_v2'] = torch.LongTensor([CATEGORY_ID_TO_SECID[self.data[idx]['category_id']]])
        return pairs_text, pairs_mask, pairs_segment, video, video_mask, labels