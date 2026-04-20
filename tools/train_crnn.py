import os
import time
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F

from ocr.dataset.yitu_dataset import YITUDataset, alphabet_encoding
from ocr.utils.pre_post_process import ctc_decode_raw
from ocr.model.crnn import CRNN
from ocr.utils.sclite_helper import ScliteHelper
from ocr.utils.warmup_scheduler import GradualWarmupScheduler
from ocr.utils.lr_scheduler import CosineAnnealingLR
from ocr.utils.xer import get_cer, get_wer
from ocr.config.defaults import _C as cfg

np.seterr(all='raise')

def val_net(dataloader):
    net.eval()
    refs = []
    hyps = []
    with torch.no_grad():
        # for i, (image, text, label_lens) in enumerate(tqdm((val_dataloader))):
        for i, (image, text, label_lens) in enumerate((dataloader)):
            if torch.cuda.is_available():
                image = image.cuda()
            else:
                image = image.to(device)

            # # B,T,C
            character_probabilities = net(image)

            arg_maxs = character_probabilities.argmax(dim=2).cpu().numpy()
            decoded_texts = ctc_decode_raw(arg_maxs, alphabet_encoding=alphabet_encoding)
            for index, decoded_text in enumerate(decoded_texts):
                actual_text = ''.join([alphabet_encoding[char_index-1] if char_index != -1 else '' for char_index in text[index].numpy()])
                refs.append(actual_text)
                hyps.append(decoded_text)
    # sclite.add_text(hyps, refs)
    # cer = sclite.get_cer()
    # wer = sclite.get_wer()
    cer = get_cer(hyps, refs)*100
    wer = get_wer(hyps, refs)*100
    return cer, wer

def train_net(epoch):
    net.train()
    for i, (images, labels, label_lens) in enumerate(train_dataloader):
        # start = time.time()
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
            label_lens = label_lens.cuda()
        else:
            images = images.to(device)
            labels = labels.to(device)
            label_lens = label_lens.to(device)

        pred = net(images)
        pred = F.log_softmax(pred, dim=-1)
        pred = pred.permute((1,0,2))  # B,T,C --> T,B,C
        # print(output.shape, labels.shape, input_lengths.shape, target_lengths.shape)
        loss = criterion(pred, labels, input_lengths, label_lens) / cfg.SOLVER.BATCH_SIZE
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.SOLVER.CLIP_GRAD)
        optimizer.step()
        scheduler.step()
        # break
        if i % cfg.SOLVER.PRINT_FREQ == 0:
            print(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            'Epoch/Iter: [{epoch}/{iter}][{i}/{length}]\t'
                'Learning rate: {lr:.6f}\t'
                # 'Speed: {lr:.2f}\t'
                'Loss: {loss:.4f}'.format(
                    epoch=epoch, iter=cfg.SOLVER.EPOCHS, i=i, length=len(train_dataset)//cfg.SOLVER.BATCH_SIZE,
                    lr=scheduler.get_lr()[0],
                    # speed=(time.time()-start)/cfg.SOLVER.BATCH_SIZE,
                    loss=loss.item()
                )
            )
            # break
        # break

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="OCR")
    parser.add_argument(
        "--cfg_file",
        default="./configs/config.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument('--gpu_id', default='0', help='gpu id')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    print(cfg)

    print(args.gpu_id)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # device_ids = []
    # ll = args.gpu_id.split(",")
    # for id in ll:
    #     device_ids.append(int(id))
    #
    # if len(device_ids) <= 0:
    #     device = torch.device('cpu')
    # elif not torch.cuda.is_available():
    #     device = torch.device('cpu')
    #     device_ids.clear()
    # else:
    #     device = torch.device('cuda:' + args.gpu_id)
    #

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # sclite = ScliteHelper()

    net = CRNN(charset_size=len(alphabet_encoding) + 1, backbone=cfg.MODEL.BACKBONE,
                         encoder_type=cfg.MODEL.ENCODER_TYPE, encoder_input_size=512,
                         encoder_hidden_size=cfg.MODEL.ENCODER_HIDDEN_SIZE,
                         encoder_layers=cfg.MODEL.ENCODER_LAYERS,
                         encoder_bidirectional=cfg.MODEL.ENCODER_BIDIRECTIONAL,
                         max_seq_len=cfg.MODEL.MAX_SEQ_LEN, ).to(device)

    if cfg.SOLVER.PRETRAINED_MODEL != '':
        print("Load pretrained model from: {}".format(cfg.SOLVER.PRETRAINED_MODEL))
        net.load_state_dict(torch.load(cfg.SOLVER.PRETRAINED_MODEL,map_location=device))

    #net = nn.DataParallel(net, device_ids=device_ids)
    #if torch.cuda.is_available():
    #    net = nn.DataParallel(net)
    #    net = net.cuda()
    # else:
    #     net = net.to(device)

    train_dataset = YITUDataset(cfg=cfg, mode='train')
    print("Number of training samples: {}".format(len(train_dataset)))
    val_dataset = YITUDataset(cfg=cfg, mode='val')
    print("Number of valing samples: {}".format(len(val_dataset)))

    # cfg.SOLVER.BATCH_SIZE = 1
    train_dataloader = data.DataLoader(train_dataset, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, num_workers=cfg.DATALOADER.NUM_WORKERS, drop_last=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False, num_workers=8)
    if cfg.SOLVER.OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-6)
    elif cfg.SOLVER.OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=1e-6, weight_decay=1e-4)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=cfg.SOLVER.LEARNING_RATE/1e-6, total_epoch=cfg.SOLVER.WARMUP_EPOCHS*len(train_dataset)//cfg.SOLVER.BATCH_SIZE,
                                       after_scheduler=CosineAnnealingLR(optimizer, cfg.SOLVER.LR_PERIOD*len(train_dataset)//cfg.SOLVER.BATCH_SIZE, eta_min=1e-6))
    """
    if (reduction == Reduction::Mean) {
        auto target_lengths_t =
            at::tensor(target_lengths, res.options()).clamp_min(1);
        return (res / target_lengths_t).mean();
    } else if (reduction == Reduction::Sum) {
        return res.sum();
    }
    """
    criterion = torch.nn.CTCLoss(blank=0, reduction='sum')
    best_cer = 1000
    input_lengths = torch.full(size=(cfg.SOLVER.BATCH_SIZE,), fill_value=cfg.MODEL.MAX_SEQ_LEN, dtype=torch.long)
    target_lengths = torch.full(size=(cfg.SOLVER.BATCH_SIZE,), fill_value=cfg.MODEL.MAX_SEQ_LEN, dtype=torch.long)
    for e in range(cfg.SOLVER.EPOCHS):
        train_net(e)

        cer, wer = val_net(val_dataloader)
        print('val current cer {:.6f}'.format(cer), 'current wer {:.6f}'.format(wer))
        if cer < best_cer:
            print("Saving network successfully, previous best cer {:.6f}, current best cer {:.6f}".format(best_cer, cer))
            best_cer = cer
            torch.save(net.state_dict(), os.path.join(cfg.SOLVER.CHECKPOINT_DIR, cfg.SOLVER.CHECKPOINT_NAME))
