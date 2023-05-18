from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import numpy as np
import random
import os
from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
import time
import argparse
import pickle
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIP4Clip, ConcatNet
from modules.module_hmcn import hmcn
from modules.optimization import BertAdam
from modules.module_loss import PolyLoss
from cn_clip.clip import _tokenizer as tokenizer
from utils.pseudo_labeling_util import pseudo_labeling 
from util import parallel_apply, get_logger
from sklearn.metrics import f1_score, accuracy_score
from category_id_map import lv2id_to_lv1id
from dataloaders.data_dataloaders import DATALOADER_DICT

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12375'
torch.distributed.init_process_group(backend="nccl")

global logger

def get_args(description='CLIP4Clip on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")

    parser.add_argument('--train_csv', type=str, default='data/.train.csv', help='')
    parser.add_argument('--val_csv', type=str, default='data/.val.csv', help='')
    parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=128, help='')
    parser.add_argument('--max_frames', type=int, default=32, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

    parser.add_argument('--loose_type', action='store_true', help="Default using tight type for retrieval.")
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")

    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--slice_framepos', type=int, default=0, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="meanP",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
                        help="choice a similarity header.")

    parser.add_argument("--pretrained_clip_name", default="clip_cn_vit-b-16.pt", type=str, help="Choose a CLIP version")
    parser.add_argument("--res", default=None, type=str, help="Enter a resume model path")
    parser.add_argument("--save_epoch", default=1, type=int, help="Save model every epoch")
    parser.add_argument('--modal_dropout', type=float, default=0.2, help='Modal Dropout Prob.')
    parser.add_argument('--epsilon', type=float, default=0.8, help='The epsilon of Poly Loss.')
    parser.add_argument('--gamma', type=float, default=0.2, help='The gamma of cal loss.')
    parser.add_argument('--do_finetune', action= 'store_true', help="Whether to run finetuning.")
    parser.add_argument('--load_finetune', type=str, default=None, help="Load finetune model.")
    parser.add_argument('--do_ssl', action='store_true', help="Whether to run semi-supervised learning.")
    parser.add_argument('--start_itr', type=int, default=0, help="Start iteration.")
    parser.add_argument('--itr', type=int, default=10, help="Total iterations.")
    parser.add_argument('--tau_p', default=0.70, type=float,
                        help='confidece threshold for positive pseudo-labels, default 0.70')
    parser.add_argument('--tau_n', default=0.05, type=float,
                        help='confidece threshold for negative pseudo-labels, default 0.05')
    parser.add_argument('--kappa_p', default=0.05, type=float,
                        help='uncertainty threshold for positive pseudo-labels, default 0.05')
    parser.add_argument('--kappa_n', default=0.005, type=float,
                        help='uncertainty threshold for negative pseudo-labels, default 0.005')
    parser.add_argument('--temp_nl', default=2.0, type=float,
                        help='temperature for generating negative pseduo-labels, default 2.0')
    parser.add_argument('--no_uncertainty', action='store_true', help="Whether to use uncertainty.")
    parser.add_argument('no_progress', action='store_true', help="Whether to use progress bar.")
    args = parser.parse_args()

    if args.sim_header == "tightTransf":
        args.loose_type = False

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    # if not args.do_train and not args.do_eval:
    #     raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    rank = torch.distributed.get_rank()
    args.rank = rank

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args

def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))
    args.device = device
    return device, n_gpu

def init_model(args, device, n_gpu, local_rank):

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)
    if args.res:
        model_state_dict = torch.load(args.res, map_location='cpu')
        if args.local_rank == 0:
                logger.info("Model loaded from %s", args.res)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model.load_state_dict(model_state_dict)

        # model.to(device)
        # return model
    if args.do_train:
        model.to(device)    

    return model

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):
    if args.do_finetune:
        model_h = hmcn(args)
        # model_h.to(device)
        model_c = ConcatNet(model, model_h, args)
        model_c.to(device)
        model = model_c
        if args.load_finetune:
            model_state_dict = torch.load(args.load_finetune, map_location='cpu')
            logger.info("Model loaded from %s", args.load_finetune)
            model.load_state_dict(state_dict=model_state_dict)
    if hasattr(model, 'module'):
        model = model.module
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', "ln"]

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]
    if args.do_finetune:
        weight_decay = 0.2
    else:
        weight_decay = 0.001
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)

       
    return optimizer, scheduler, model

def save_model(epoch, args, model, optimizer, tr_loss, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", epoch+1))
    optimizer_state_file = os.path.join(
        args.output_dir, "pytorch_opt.bin.{}{}".format("" if type_name=="" else type_name+".", epoch+1))
    torch.save(model_to_save.state_dict(), output_model_file)
    torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': tr_loss,
            }, optimizer_state_file)
    logger.info("Model saved to %s", output_model_file)
    logger.info("Optimizer saved to %s", optimizer_state_file)
    return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)
    else:
        model = None
    return model

def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, video, video_mask = batch
        loss = model(input_ids, segment_ids, input_mask, video, video_mask)

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()

            # https://github.com/openai/CLIP/issues/46
            if hasattr(model, 'module'):
                torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            else:
                torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.9f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step

def finetune_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0
    total_accuracy = 0
    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, video, video_mask, groud_truth = batch
        loss, accuracy, pred_label_id, label = model(input_ids, segment_ids, input_mask, video, video_mask, groud_truth)

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        total_loss += float(loss)
        total_accuracy += float(accuracy)
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()

            # # https://github.com/openai/CLIP/issues/46
            # if hasattr(model, 'module'):
            #     torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            # else:
            #     torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Accuracy: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.9f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]), 
                            float(loss), float(accuracy),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()
    
    total_loss = total_loss / len(train_dataloader)
    total_accuracy = total_accuracy / len(train_dataloader)
    return total_loss, total_accuracy, global_step

def _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list):
    sim_matrix = []
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, segment_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        each_row = []
        for idx2, b2 in enumerate(batch_list_v):
            video_mask, *_tmp = b2
            visual_output = batch_visual_output_list[idx2]
            b1b2_logits, *_tmp = model.get_similarity_logits(sequence_output, visual_output, input_mask, video_mask,
                                                                     loose_type=model.loose_type)
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    return sim_matrix

def eval_epoch(args, model, test_dataloader, device, n_gpu):

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # #################################################################
    ## below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per video clip setting.")
        logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        total_video_num = 0

        # ----------------------------
        # 1. cache the features
        # ----------------------------
        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, video, video_mask = batch

            if multi_sentence_:
                # multi-sentences retrieval means: one clip has two or more descriptions.
                b, *_t = video.shape
                sequence_output = model.get_sequence_output(input_ids, segment_ids, input_mask)
                batch_sequence_output_list.append(sequence_output)
                batch_list_t.append((input_mask, segment_ids,))

                s_, e_ = total_video_num, total_video_num + b
                filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

                if len(filter_inds) > 0:
                    video, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]
                    visual_output = model.get_visual_output(video, video_mask)
                    batch_visual_output_list.append(visual_output)
                    batch_list_v.append((video_mask,))
                total_video_num += b
            else:
                sequence_output, visual_output = model.get_sequence_visual_output(input_ids, segment_ids, input_mask, video, video_mask)

                batch_sequence_output_list.append(sequence_output)
                batch_list_t.append((input_mask, segment_ids,))

                batch_visual_output_list.append(visual_output)
                batch_list_v.append((video_mask,))

            print("{}/{}\r".format(bid, len(test_dataloader)), end="")

        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------
        if n_gpu > 1:
            device_ids = list(range(n_gpu))
            batch_list_t_splits = []
            batch_list_v_splits = []
            batch_t_output_splits = []
            batch_v_output_splits = []
            bacth_len = len(batch_list_t)
            split_len = (bacth_len + n_gpu - 1) // n_gpu
            for dev_id in device_ids:
                s_, e_ = dev_id * split_len, (dev_id + 1) * split_len
                if dev_id == 0:
                    batch_list_t_splits.append(batch_list_t[s_:e_])
                    batch_list_v_splits.append(batch_list_v)

                    batch_t_output_splits.append(batch_sequence_output_list[s_:e_])
                    batch_v_output_splits.append(batch_visual_output_list)
                else:
                    devc = torch.device('cuda:{}'.format(str(dev_id)))
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_t[s_:e_]]
                    batch_list_t_splits.append(devc_batch_list)
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_v]
                    batch_list_v_splits.append(devc_batch_list)

                    devc_batch_list = [b.to(devc) for b in batch_sequence_output_list[s_:e_]]
                    batch_t_output_splits.append(devc_batch_list)
                    devc_batch_list = [b.to(devc) for b in batch_visual_output_list]
                    batch_v_output_splits.append(devc_batch_list)

            parameters_tuple_list = [(batch_list_t_splits[dev_id], batch_list_v_splits[dev_id],
                                      batch_t_output_splits[dev_id], batch_v_output_splits[dev_id]) for dev_id in device_ids]
            parallel_outputs = parallel_apply(_run_on_single_gpu, model, parameters_tuple_list, device_ids)
            sim_matrix = []
            for idx in range(len(parallel_outputs)):
                sim_matrix += parallel_outputs[idx]
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
        else:
            sim_matrix = _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list)
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

    if multi_sentence_:
        logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                                                  np.full((max_length-e_+s_, sim_matrix.shape[1]), -np.inf)), axis=0))
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logger.info("after reshape, sim matrix size: {} x {} x {}".
                    format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))

        tv_metrics = tensor_text_to_video_metrics(sim_matrix)
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))
    else:
        logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        tv_metrics = compute_metrics(sim_matrix)
        vt_metrics = compute_metrics(sim_matrix.T)
        logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

    logger.info("Text-to-Video:")
    logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
    logger.info("Video-to-Text:")
    logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))

    R1 = tv_metrics['R1']
    return R1

def eval_fine_epoch(args, model, eval_dataloader, device, n_gpu, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.eval()
    log_step = args.n_display
    start_time = time.time()
    all_pred_label_ids = torch.tensor([], dtype=torch.long, device=device)
    all_label = torch.tensor([], dtype=torch.long, device=device)
    
    for step, batch in enumerate(eval_dataloader):
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, video, video_mask, groud_truth = batch
        loss, accuracy, pred_label_id, label = model(input_ids, segment_ids, input_mask, video, video_mask, groud_truth)
        all_pred_label_ids = torch.cat((all_pred_label_ids, pred_label_id), dim=0)
        all_label = torch.cat((all_label, label), dim=0)
    # print(all_label.shape)
    # print(all_pred_label_ids.shape)
    all_label = all_label.cpu().numpy()
    # print(all_label.shape)
    all_pred_label_ids = all_pred_label_ids.cpu().numpy()
    # print(all_pred_label_ids.shape)
    # y_pred = lv2id_to_lv1id(all_pred_label_ids)
    y_pred = list(map(lv2id_to_lv1id, all_pred_label_ids.tolist()))
    # print(len(y_pred))
    # y_true = lv2id_to_lv1id(all_label)
    y_true = list(map(lv2id_to_lv1id, all_label.tolist()))
    # print(len(y_true))
    F1_lv1 = (f1_score(y_true, y_pred, average='macro')+f1_score(y_true, y_pred, average='micro'))/2
    accu_lv1 = accuracy_score(y_true, y_pred)
    F1_lv2 = (f1_score(all_label, all_pred_label_ids, average='macro')+f1_score(all_label, all_pred_label_ids, average='micro'))/2
    accu_lv2 = accuracy_score(all_label, all_pred_label_ids)
    F1_score = (F1_lv1+F1_lv2)/2
    return F1_lv1, F1_lv2, F1_score, accu_lv1, accu_lv2

def ssl_train(args, model, lbl_loader, nl_loader, device, n_gpu, optimizer, scheduler, global_step, itr=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0
    total_accuracy = 0
    
    train_loader = zip(lbl_loader, nl_loader)
    
    for step, (batch_x, batch_nl) in enumerate(train_loader):
        # batch_x = tuple(t.to(device=device, non_blocking=True) for t in batch_x)
        # batch_nl = tuple(t.to(device=device, non_blocking=True) for t in batch_nl)
        input_ids_x, input_mask_x, segment_ids_x, video_x, video_mask_x, groud_truth_x, _, nl_mask_x = batch_x
        input_ids_nl, input_mask_nl, segment_ids_nl, video_nl, video_mask_nl, groud_truth_nl, _, nl_mask_nl = batch_nl
        input_ids = torch.cat((input_ids_x, input_ids_nl)).to(device)
        input_mask = torch.cat((input_mask_x, input_mask_nl)).to(device)
        segment_ids = torch.cat((segment_ids_x, segment_ids_nl)).to(device)
        video = torch.cat((video_x, video_nl)).to(device)
        video_mask = torch.cat((video_mask_x, video_mask_nl)).to(device)
        # print(groud_truth_x)
        nl_mask = torch.cat((nl_mask_x, nl_mask_nl)).to(device)
        # groud_truth = np.array(list(map(int, groud_truth_x))+list(map(int, groud_truth_nl)))
        # groud_truth = torch.from_numpy(groud_truth).to(device)
        logits = model(input_ids, segment_ids, input_mask, video, video_mask)

        positive_idx = nl_mask.sum(dim=1) == 200 #the mask for negative learning is all ones
        positive_idx = positive_idx.cpu().numpy()
        ground_truth = torch.cat((groud_truth_x, groud_truth_nl)).to(device)
        ground_truth = ground_truth[positive_idx].squeeze()
        # print(nl_mask.shape)
        # print((nl_mask.sum(dim=1) != 200))
        # print((nl_mask.sum(dim=1) > 0))
        trig = sum((nl_mask.sum(dim=1) != 200) * (nl_mask.sum(dim=1) > 0) * 1)
        nl_idx = (nl_mask.sum(dim=1) != 200) * (nl_mask.sum(dim=1) > 0)
        
        loss_ce = 0
        loss_nl = 0
        
        if sum(positive_idx) > 0:
            loss_ce = torch.nn.functional.cross_entropy(logits[positive_idx], ground_truth)
            # print(ground_truth)
            # t, _, _, _ = model(input_ids[positive_idx], segment_ids[positive_idx], input_mask[positive_idx], video[positive_idx], video_mask[positive_idx], ground_truth)
            # loss_ce += t
        # print(ground_truth.shape)
        # print(loss_ce)
        if trig > 0:
            
            nl_logits = logits[nl_idx]
            
            pred_nl = torch.nn.functional.softmax(nl_logits, dim=1)
            pred_nl = 1 - pred_nl
            pred_nl = torch.clamp(pred_nl, min=1e-7, max=1.0)
            nl_mask = nl_mask[nl_idx]
            y_nl = torch.ones((nl_logits.shape)).to(device=args.device, dtype=logits.dtype)
            loss_nl += torch.mean((-torch.sum((y_nl * torch.log(pred_nl))*nl_mask, dim = -1))/(torch.sum(nl_mask, dim = -1) + 1e-7))
            
        loss = loss_ce + loss_nl
        loss.backward()
        if scheduler is not None:
            scheduler.step()  # Update learning rate schedule

        optimizer.step()
        optimizer.zero_grad()
        
        global_step += 1
        # if global_step % log_step == 0:
        #     logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
        #                 args.epochs, step + 1,
        #                 len(train_dataloader), "-".join([str('%.9f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]), 
        #                 float(loss), 
        #                 (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
        #     start_time = time.time()



def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    
    args.local_device_rank = max(args.local_rank, 0)
    device, n_gpu = init_device(args, args.local_rank)

    # tokenizer = FullTokenizer()

    assert  args.task_type == "retrieval"
    model = init_model(args, device, n_gpu, args.local_rank)

    ## ####################################
    # freeze testing
    ## ####################################
    assert args.freeze_layer_num <= 12 and args.freeze_layer_num >= -1
    if hasattr(model, "clip") and args.freeze_layer_num > -1:
        for name, param in model.clip.named_parameters():
            # top layers always need to train
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                    or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                continue    # need to train
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= args.freeze_layer_num:
                    continue    # need to train

            if args.linear_patch == "3d" and name.find("conv2."):
                continue
            else:
                # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False

    ## ####################################
    # dataloader loading
    ## ####################################
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None \
           or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length, _ = DATALOADER_DICT[args.datatype]["val"](args, tokenizer)
    else:
        val_dataloader, val_length = test_dataloader, test_length

    ## report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    if args.local_rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)

    ## ####################################
    # train and eval
    ## ####################################
    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        # print("train length: {}".format(len(train_dataloader)))
        # exit(1)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        coef_lr = args.coef_lr
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_score = 0.00001
        best_output_model_file = "None"
        ## ##############################################################
        # resume optimizer state besides loss to continue train
        ## ##############################################################
        resumed_epoch = 0
        if args.resume_model:
            checkpoint = torch.load(args.resume_model, map_location='cpu')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            resumed_epoch = checkpoint['epoch']+1
            resumed_loss = checkpoint['loss']
        
        global_step = 0
        for epoch in range(resumed_epoch, args.epochs):
            if args.res is None and epoch==0:
            #     # Freeze all parameters in the CLIP model
            #     for param in model.parameters():
            #         # print(param)
            #         param.requires_grad = False

                # Unfreeze the parameters in the ResidualMLP
                for name, param in model.named_parameters():
                    if(name.startswith("clip")):
                        # print(name)
                        param.requires_grad = False
                # exit(1)
            NoMem = True
            while(NoMem):
                try:
                    train_sampler.set_epoch(epoch)
                    tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                               scheduler, global_step, local_rank=args.local_rank)
                except:
                    torch.cuda.empty_cache()
                    logger.info("Out of memory, retrying...")
                    time.sleep(100)
                else:
                    NoMem = False
            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
                if(epoch%args.save_epoch==0):
                    output_model_file = save_model(epoch, args, model, optimizer, tr_loss, type_name="large scale")

                ## Run on val dataset, this process is *TIME-consuming*.
                # logger.info("Eval on val dataset")
                # R1 = eval_epoch(args, model, val_dataloader, device, n_gpu)

                # R1 = eval_epoch(args, model, test_dataloader, device, n_gpu)
                # if best_score <= R1:
                #     best_score = R1
                #     best_output_model_file = output_model_file
                # logger.info("The best model is: {}, the R1 is: {:.4f}".format(best_output_model_file, best_score))
            if args.res is None and epoch==0:
                # Unfreeze all parameters in the CLIP model
                if hasattr(model, "clip") and args.freeze_layer_num > -1:
                    for name, param in model.clip.named_parameters():
                # top layers always need to train
                        if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                            or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                            continue    # need to train
                        elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                            layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                            if layer_num >= args.freeze_layer_num:
                                continue    # need to train

                        if args.linear_patch == "3d" and name.find("conv2."):
                            continue
                        else:
                        # paramenters which < freeze_layer_num will be freezed
                            param.requires_grad = False
                else:
                    for param in model.parameters():
                        param.requires_grad = True
        ## Uncomment if want to test on the best checkpoint
        # if args.local_rank == 0:
        #     model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file)
        #     eval_epoch(args, model, test_dataloader, device, n_gpu)
    elif args.do_finetune:
        if args.do_ssl:
            start_itr = args.start_itr
            coef_lr = args.coef_lr
            optimizer, scheduler, model = prep_optimizer(args, model, -1, device, n_gpu, args.local_rank, coef_lr=coef_lr)
            unlbl_loader, _ = DATALOADER_DICT[args.datatype]["unlbl"](args, tokenizer)
            for itr in range(start_itr, args.itr):
                
                test_dataloader,test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

                if os.path.exists(f'{args.output_dir}/pseudo_labeling_iteration_{str(itr)}.pkl'):
                    pseudo_label_dict = pickle.load(open(f'{args.output_dir}/pseudo_labeling_iteration_{str(itr)}.pkl', 'rb'))
                else:
                    unique_sel_neg, pseudo_label_dict = pseudo_labeling(args, unlbl_loader, model, itr)
                    with open(os.path.join(args.output_dir, f'pseudo_labeling_iteration_{str(itr)}.pkl'),"wb") as f:
                        pickle.dump(pseudo_label_dict,f)
                        logger.info(f"Save pseudo_labeling_iteration_{str(itr)}.pkl")
                lbl_loader, nl_loader = DATALOADER_DICT[args.datatype]["ssl"](args, tokenizer, pseudo_label_dict)
                
                resumed_epoch = 0
                global_step = 0
        
                args.freeze_layer_num = 10
                vis = True
                for name, param in model.module.net1.named_parameters():
                    if name.find("ln_final.") == 0 or name.find("clip.text_projection") == 0 or name.find("clip.logit_scale") == 0 \
                        or name.find("ln_4.") == 0 or name.find("visual.proj") == 0:
                            continue
                    if name.find("clip.bert.encoder.layer.") == 0:
                        layer_num = int(name.split(".")[4])
                        if layer_num >= args.freeze_layer_num:
                            param.requires_grad = True
                            continue
                    if name.find("clip") == 0:
                        param.requires_grad = False
                        print(name)
                    else:
                        param.requires_grad = True

                for epoch in range(resumed_epoch, args.epochs):
                    ssl_train(args, model, lbl_loader, nl_loader, device, n_gpu, optimizer, scheduler, global_step, itr=itr)
                    if args.local_rank == 0:
                        logger.info("Epoch %d/%s Finished", epoch + 1, args.epochs)
                        F1_lv1, F1_lv2, F1_score, accu_lv1, accu_lv2 = eval_fine_epoch(args, model, test_dataloader, device, n_gpu,
                                                global_step, local_rank=args.local_rank) 
                        # if F1_score > best_score:
                        logger.info("The F1 is: {:.4f}, the Lv1 F1 is: {:.4f}, the Lv2 F1 is {:.4f}, the Accuracy is {:.4f}, the Lv1 Accuracy is {:.4f}".format(F1_score, F1_lv1, F1_lv2, accu_lv2, accu_lv1))
                        if (epoch+1)%args.save_epoch == 0:
                            output_model_file = save_model(epoch, args, model, optimizer, tr_loss, type_name="ssl_{}".format(itr))
                    
        else:
            train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["val"](args, tokenizer)
            test_dataloader,test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)
            num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                            / args.gradient_accumulation_steps) * args.epochs

            coef_lr = args.coef_lr
            optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)
            
            if args.local_rank == 0:
                logger.info("***** Running training *****")
                logger.info("  Num examples = %d", train_length)
                logger.info("  Batch size = %d", args.batch_size)
                logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

            best_score = 0.00001
            best_output_model_file = "None"
            ## ##############################################################
            # resume optimizer state besides loss to continue train
            ## ##############################################################
            resumed_epoch = 0
            if args.resume_model:
                checkpoint = torch.load(args.resume_model, map_location='cpu')
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                resumed_epoch = checkpoint['epoch']+1
                resumed_loss = checkpoint['loss']
            
            global_step = 0
            best_score = 0.00001
            for epoch in range(resumed_epoch, args.epochs):
                if epoch == 0 or epoch == 20 or epoch == resumed_epoch:
                    vis = False
                    if epoch <20:
                        args.freeze_layer_num = 12
                    else:
                        args.freeze_layer_num = 10
                        vis = True
                    for name, param in model.module.net1.named_parameters():
                        if name.find("ln_final.") == 0 or name.find("clip.text_projection") == 0 or name.find("clip.logit_scale") == 0 \
                            or name.find("ln_4.") == 0 or name.find("visual.proj") == 0:
                                continue
                        if name.find("clip.bert.encoder.layer.") == 0:
                            layer_num = int(name.split(".")[4])
                            if layer_num >= args.freeze_layer_num:
                                param.requires_grad = True
                                continue
                        if name.find("clip") == 0:
                            param.requires_grad = False
                            print(name)
                        else:
                            param.requires_grad = True

                train_sampler.set_epoch(epoch)
                tr_loss, tr_accu, global_step = finetune_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                        scheduler, global_step, local_rank=args.local_rank)
                if args.local_rank == 0:
                    logger.info("Epoch %d/%s Finished, Train Loss: %f, Train Accuracy: %f", epoch + 1, args.epochs, tr_loss, tr_accu)
                    F1_lv1, F1_lv2, F1_score, accu_lv1, accu_lv2 = eval_fine_epoch(args, model, test_dataloader, device, n_gpu,
                                            global_step, local_rank=args.local_rank) 
                    # if F1_score > best_score:
                    logger.info("The model is: {}, the F1 is: {:.4f}, the Lv1 F1 is: {:.4f}, the Accuracy is {:.4f}, the Lv1 Accuracy is {:.4f}".format(output_model_file, F1_score, F1_lv1, accu_lv2, accu_lv1))
                    if (epoch+1)%args.save_epoch == 0:
                        output_model_file = save_model(epoch, args, model, optimizer, tr_loss, type_name="finetuning")
                        # best_score = F1_score
                        
    elif args.do_eval:
        if args.local_rank == 0:
            eval_epoch(args, model, test_dataloader, device, n_gpu)
if __name__ == "__main__":
    main()
