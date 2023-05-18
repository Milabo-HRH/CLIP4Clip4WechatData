import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_DataLoader
from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_TrainDataLoader
from dataloaders.dataloader_msvd_retrieval import MSVD_DataLoader
from dataloaders.dataloader_lsmdc_retrieval import LSMDC_DataLoader
from dataloaders.dataloader_activitynet_retrieval import ActivityNet_DataLoader
from dataloaders.dataloader_didemo_retrieval import DiDeMo_DataLoader
from dataloaders.dataloader_wechat_retrieval import Wechat_DataLoader, Wechat_Dataloader_finetune, Wechat_Dataloader_eval, Wechat_Dataloader_ssl, Wechat_DataLoader_unlbl
import os
import numpy as np
def dataloader_msrvtt_train(args, tokenizer):
    msrvtt_dataset = MSRVTT_TrainDataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=args.expand_msrvtt_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler

def dataloader_msrvtt_test(args, tokenizer, subset="test"):
    msrvtt_testset = MSRVTT_DataLoader(
        csv_path=args.val_csv,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)


def dataloader_msvd_train(args, tokenizer):
    msvd_dataset = MSVD_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msvd_dataset)
    dataloader = DataLoader(
        msvd_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msvd_dataset), train_sampler

def dataloader_msvd_test(args, tokenizer, subset="test"):
    msvd_testset = MSVD_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        msvd_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msvd_testset)


def dataloader_lsmdc_train(args, tokenizer):
    lsmdc_dataset = LSMDC_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(lsmdc_dataset)
    dataloader = DataLoader(
        lsmdc_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(lsmdc_dataset), train_sampler

def dataloader_lsmdc_test(args, tokenizer, subset="test"):
    lsmdc_testset = LSMDC_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        lsmdc_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(lsmdc_testset)


def dataloader_activity_train(args, tokenizer):
    activity_dataset = ActivityNet_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(activity_dataset)
    dataloader = DataLoader(
        activity_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(activity_dataset), train_sampler

def dataloader_activity_test(args, tokenizer, subset="test"):
    activity_testset = ActivityNet_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        activity_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(activity_testset)


def dataloader_didemo_train(args, tokenizer):
    didemo_dataset = DiDeMo_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(didemo_dataset)
    dataloader = DataLoader(
        didemo_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(didemo_dataset), train_sampler

def dataloader_didemo_test(args, tokenizer, subset="test"):
    didemo_testset = DiDeMo_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_didemo = DataLoader(
        didemo_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_didemo, len(didemo_testset)

def dataloader_Wechat_train(args, tokenizer):
    wechat_dataset = Wechat_DataLoader(
        subset="train",
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        ocr = False,
        num_thread_reader = args.num_thread_reader,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(wechat_dataset)
    dataloader = DataLoader(
        wechat_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=4,
        pin_memory=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(wechat_dataset), train_sampler

def dataloader_Wechat_finetune(args, tokenizer, subset='val'):
    wechat_dataset = Wechat_Dataloader_finetune(
        subset="val",
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        ocr = False,
        num_thread_reader = args.num_thread_reader,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(wechat_dataset)
    dataloader = DataLoader(
        wechat_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=4,
        pin_memory=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(wechat_dataset), train_sampler

def dataloader_Wechat_test(args, tokenizer, subset="test"):
    wechat_testset = Wechat_Dataloader_eval(
        subset="val",
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        ocr = False,
        num_thread_reader = args.num_thread_reader,
    )
    dataloader_wechat = DataLoader(
        wechat_testset,
        batch_size=args.batch_size_val,
        num_workers=4,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_wechat, len(wechat_testset)

def dataloader_Wechat_ssl(args, tokenizer, pseudo_lbl_dict, itr=0):
    lbl_idx = list(range(90000))
    train_unlbl_idx = list(range(90000, 1090000))
    # pseudo_lbl_dict = pickle.load(open(pseudo_lbl, 'rb'))
    pseudo_idx = pseudo_lbl_dict['pseudo_idx']
    pseudo_target = pseudo_lbl_dict['pseudo_target']
    nl_idx = pseudo_lbl_dict['nl_idx']
    nl_mask = pseudo_lbl_dict['nl_mask']
    lbl_idx = np.array(lbl_idx + pseudo_idx)

    #balance the labeled and unlabeled data 
    if len(nl_idx) > len(lbl_idx):
        exapand_labeled = len(nl_idx) // len(lbl_idx)
        lbl_idx = np.hstack([lbl_idx for _ in range(exapand_labeled)])

        if len(lbl_idx) < len(nl_idx):
            diff = len(nl_idx) - len(lbl_idx)
            lbl_idx = np.hstack((lbl_idx, np.random.choice(lbl_idx, diff)))
        else:
            assert len(lbl_idx) == len(nl_idx)

    train_lbl_dataset = Wechat_Dataloader_ssl(
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        indexs=lbl_idx,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        ocr = False,
        pseudo_idx=pseudo_idx,
        pseudo_target=pseudo_target,
        nl_idx=nl_idx,
        nl_mask=nl_mask,
        num_thread_reader = args.num_thread_reader,
    )
    train_nl_dataset = Wechat_Dataloader_ssl(
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        indexs=np.array(nl_idx),
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        ocr = False,
        pseudo_idx=pseudo_idx,
        pseudo_target=pseudo_target,
        nl_idx=nl_idx,
        nl_mask=nl_mask,
        num_thread_reader = args.num_thread_reader,
    )
    lbl_loader = DataLoader(
        train_lbl_dataset,
        sampler=RandomSampler(train_lbl_dataset),
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=True)

    nl_loader = DataLoader(
        train_nl_dataset,
        sampler=RandomSampler(train_nl_dataset),
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=True)
    return lbl_loader, nl_loader

def dataloader_Wechat_unlbl(args, tokenizer):
    train_unlbl_dataset = Wechat_DataLoader_unlbl(
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        ocr = False,
        num_thread_reader = args.num_thread_reader,
    )
    dataloader_wechat = DataLoader(
        train_unlbl_dataset,
        batch_size=args.batch_size_val,
        num_workers=4,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_wechat, len(train_unlbl_dataset)

DATALOADER_DICT = {}
DATALOADER_DICT["msrvtt"] = {"train":dataloader_msrvtt_train, "val":dataloader_msrvtt_test, "test":None}
DATALOADER_DICT["msvd"] = {"train":dataloader_msvd_train, "val":dataloader_msvd_test, "test":dataloader_msvd_test}
DATALOADER_DICT["lsmdc"] = {"train":dataloader_lsmdc_train, "val":dataloader_lsmdc_test, "test":dataloader_lsmdc_test}
DATALOADER_DICT["activity"] = {"train":dataloader_activity_train, "val":dataloader_activity_test, "test":None}
DATALOADER_DICT["didemo"] = {"train":dataloader_didemo_train, "val":dataloader_didemo_test, "test":dataloader_didemo_test}
DATALOADER_DICT["wechat"] = {"train":dataloader_Wechat_train, "val":dataloader_Wechat_finetune, "test":dataloader_Wechat_test, "ssl": dataloader_Wechat_ssl, "unlbl": dataloader_Wechat_unlbl}