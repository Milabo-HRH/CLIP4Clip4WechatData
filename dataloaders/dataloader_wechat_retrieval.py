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
from io import BytesIO


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
    ):
        # todo1: need to preprocess the data
        # self.csv = pd.read_csv(csv_path)
        if subset == "train":   
            self.json_path = os.path.join(json_path, "train_list.txt")
        else:
            self.json_path = os.path.join(json_path, "test_list.txt")
        self.data = json.load(open(json_path, 'r'))
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
        self.num_workers = args.num_thread_reader
        if self.num_workers > 0:
            # lazy initialization for zip_handler to avoid multiprocessing-reading error
            self.handles = [None for _ in range(args.num_workers)]
        else:
            self.handles = zipfile.ZipFile(self.zip_feat_path, 'r')

        self.ocr = ocr
        train_video_ids = list(self.data['id'].values)
        self.sample_len = len(self.data)
        # if self.ocr:
        #     train_video_ids = list(self.csv['id'].values)
        #     self.sentences_dict = {}
        #     for itm in self.data['sentences']:
        #         if itm['id'] in train_video_ids:
        #             self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['caption'])
        #     self.sample_len = len(self.sentences_dict)
        # else:
        #     num_sentences = 0
        #     self.sentences = defaultdict(list)
        #     s_video_id_set = set()
        #     for itm in self.data['sentences']:
        #         self.sentences[itm['video_id']].append(itm['caption'])
        #         num_sentences += 1
        #         s_video_id_set.add(itm['video_id'])

        #     # Use to find the clips in the same video
        #     self.parent_ids = {}
        #     self.children_video_ids = defaultdict(list)
        #     for itm in self.data['videos']:
        #         vid = itm["video_id"]
        #         url_posfix = itm["url"].split("?v=")[-1]
        #         self.parent_ids[vid] = url_posfix
        #         self.children_video_ids[url_posfix].append(vid)
        #     self.sample_len = len(self.csv)

        # self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
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
                words = self.tokenizer.tokenize(data[video_id]['title'])

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

        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)
        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test_mode:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
            else:
                # randomly sample when test mode is False
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[:self.max_frame]
                select_inds = sorted(select_inds)
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        # mask = torch.LongTensor(mask)
        return feat, mask
        # video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        # max_video_length = [0] * len(choice_video_ids)

        # # Pair x L x T x 3 x H x W
        # video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
        #                   self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

        # for i, video_id in enumerate(choice_video_ids):
        #     # Individual for YoucokII dataset, due to it video format
        #     video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
        #     if os.path.exists(video_path) is False:
        #         video_path = video_path.replace(".mp4", ".webm")

        #     raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
        #     raw_video_data = raw_video_data['video']
        #     if len(raw_video_data.shape) > 3:
        #         raw_video_data_clip = raw_video_data
        #         # L x T x 3 x H x W
        #         raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
        #         if self.max_frames < raw_video_slice.shape[0]:
        #             if self.slice_framepos == 0:
        #                 video_slice = raw_video_slice[:self.max_frames, ...]
        #             elif self.slice_framepos == 1:
        #                 video_slice = raw_video_slice[-self.max_frames:, ...]
        #             else:
        #                 sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
        #                 video_slice = raw_video_slice[sample_indx, ...]
        #         else:
        #             video_slice = raw_video_slice

        #         video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

        #         slice_len = video_slice.shape[0]
        #         max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
        #         if slice_len < 1:
        #             pass
        #         else:
        #             video[i][:slice_len, ...] = video_slice
        #     else:
        #         print("video path: {} error. video id: {}".format(video_path, video_id))

        # for i, v_length in enumerate(max_video_length):
        #     video_mask[i][:v_length] = [1] * v_length

        # return video, video_mask

    def __getitem__(self, idx):
        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(idx)
        video, video_mask = self._get_rawvideo(choice_video_ids)
        return pairs_text, pairs_mask, pairs_segment, video, video_mask

    # """Wechat dataset loader."""
    # def __init__(
    #         self,
    #         subset,
    #         data_path,
    #         features_path,
    #         tokenizer,
    #         max_words=30,
    #         feature_framerate=1.0,
    #         max_frames=30,
    #         image_resolution=224,
    #         frame_order=0,
    #         slice_framepos=0,
    # ):
    #     self.data_path = data_path
    #     self.features_path = features_path
    #     self.feature_framerate = feature_framerate
    #     self.max_words = max_words
    #     self.max_frames = max_frames
    #     self.tokenizer = tokenizer
    #     # 0: ordinary order; 1: reverse order; 2: random order.
    #     self.frame_order = frame_order
    #     assert self.frame_order in [0, 1, 2]
    #     # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
    #     self.slice_framepos = slice_framepos
    #     assert self.slice_framepos in [0, 1, 2]

    #     self.subset = subset
    #     assert self.subset in ["train", "val", "test"]
    #     video_id_path_dict = {}
    #     #todo 1: process the data before training
    #     video_id_path_dict["train"] = os.path.join(self.data_path, "train_list.txt")
    #     video_id_path_dict["val"] = os.path.join(self.data_path, "val_list.txt")
    #     video_id_path_dict["test"] = os.path.join(self.data_path, "test_list.txt")
    #     caption_file = os.path.join(self.data_path, "pretrain.json")

    #     with open(video_id_path_dict[self.subset], 'r') as fp:
    #         video_ids = [itm.strip() for itm in fp.readlines()]

    #     with open(caption_file, 'r', encoding='utf8') as f:
    #         captions = json.load(f)

    #     video_dict = {}
    #     for root, dub_dir, video_files in os.walk(self.features_path):
    #         for video_file in video_files:
    #             video_id_ = ".".join(video_file.split(".")[:-1])
    #             if video_id_ not in video_ids:
    #                 continue
    #             file_path_ = os.path.join(root, video_file)
    #             video_dict[video_id_] = file_path_
    #     self.video_dict = video_dict

    #     self.sample_len = 0
    #     self.sentences_dict = {}
    #     self.cut_off_points = []
    #     for video_id in video_ids:
    #         assert video_id in captions
    #         for cap in captions[video_id]:
    #             cap_txt = " ".join(cap)
    #             self.sentences_dict[len(self.sentences_dict)] = (video_id, cap_txt)
    #         self.cut_off_points.append(len(self.sentences_dict))

    #     ## below variables are used to multi-sentences retrieval
    #     # self.cut_off_points: used to tag the label when calculate the metric
    #     # self.sentence_num: used to cut the sentence representation
    #     # self.video_num: used to cut the video representation
    #     self.multi_sentence_per_video = True    # !!! important tag for eval
    #     if self.subset == "val" or self.subset == "test":
    #         self.sentence_num = len(self.sentences_dict)
    #         self.video_num = len(video_ids)
    #         assert len(self.cut_off_points) == self.video_num
    #         print("For {}, sentence number: {}".format(self.subset, self.sentence_num))
    #         print("For {}, video number: {}".format(self.subset, self.video_num))

    #     print("Video number: {}".format(len(self.video_dict)))
    #     print("Total Paire: {}".format(len(self.sentences_dict)))

    #     self.sample_len = len(self.sentences_dict)
    #     self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
    #     self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
    #                           "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    # def __len__(self):
    #     return self.sample_len

    # def _get_text(self, video_id, caption):
    #     k = 1
    #     choice_video_ids = [video_id]
    #     pairs_text = np.zeros((k, self.max_words), dtype=np.long)
    #     pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
    #     pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

    #     for i, video_id in enumerate(choice_video_ids):
    #         words = self.tokenizer.tokenize(caption)

    #         words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
    #         total_length_with_CLS = self.max_words - 1
    #         if len(words) > total_length_with_CLS:
    #             words = words[:total_length_with_CLS]
    #         words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

    #         input_ids = self.tokenizer.convert_tokens_to_ids(words)
    #         input_mask = [1] * len(input_ids)
    #         segment_ids = [0] * len(input_ids)
    #         while len(input_ids) < self.max_words:
    #             input_ids.append(0)
    #             input_mask.append(0)
    #             segment_ids.append(0)
    #         assert len(input_ids) == self.max_words
    #         assert len(input_mask) == self.max_words
    #         assert len(segment_ids) == self.max_words

    #         pairs_text[i] = np.array(input_ids)
    #         pairs_mask[i] = np.array(input_mask)
    #         pairs_segment[i] = np.array(segment_ids)

    #     return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    # def _get_rawvideo(self, choice_video_ids):
    #     video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
    #     max_video_length = [0] * len(choice_video_ids)

    #     # Pair x L x T x 3 x H x W
    #     video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
    #                       self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

    #     for i, video_id in enumerate(choice_video_ids):
    #         video_path = self.video_dict[video_id]

    #         raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
    #         raw_video_data = raw_video_data['video']

    #         if len(raw_video_data.shape) > 3:
    #             raw_video_data_clip = raw_video_data
    #             # L x T x 3 x H x W
    #             raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
    #             if self.max_frames < raw_video_slice.shape[0]:
    #                 if self.slice_framepos == 0:
    #                     video_slice = raw_video_slice[:self.max_frames, ...]
    #                 elif self.slice_framepos == 1:
    #                     video_slice = raw_video_slice[-self.max_frames:, ...]
    #                 else:
    #                     sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
    #                     video_slice = raw_video_slice[sample_indx, ...]
    #             else:
    #                 video_slice = raw_video_slice

    #             video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

    #             slice_len = video_slice.shape[0]
    #             max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
    #             if slice_len < 1:
    #                 pass
    #             else:
    #                 video[i][:slice_len, ...] = video_slice
    #         else:
    #             print("video path: {} error. video id: {}".format(video_path, video_id))

    #     for i, v_length in enumerate(max_video_length):
    #         video_mask[i][:v_length] = [1] * v_length

    #     return video, video_mask

    # def __getitem__(self, idx):
    #     video_id, caption = self.sentences_dict[idx]

    #     pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
    #     video, video_mask = self._get_rawvideo(choice_video_ids)
    #     return pairs_text, pairs_mask, pairs_segment, video, video_mask
