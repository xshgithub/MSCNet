# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable

from models.dqdetr.matcher import HungarianMatcher
from util.utils import slprint, to_device

import torch
import torch.nn as nn
import torch.nn.functional as F
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from vis import (
    compute_per_image_ap,
    save_detection_visualization,
    visualize_topk_split_gt_pred,
)

print_freq = 100
CCM_LOSS = torch.nn.CrossEntropyLoss()
ccm_coeff = 1

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None):
    def debug_once(model, samples, targets=None):
        device = next(model.parameters()).device
        model.eval()  # 只查看值，不改变模型
        with torch.no_grad():
            # 如果你的 samples 是 NestedTensor，按原方式传入
            features, poss = model.backbone(samples)
            print("=== BACKBONE FEATURES ===")
            for i, feat in enumerate(features):
                if hasattr(feat, 'shape'):
                    t = feat
                else:
                    t = feat.tensors if hasattr(feat, 'tensors') else feat
                print(f"stage {i}: shape={t.shape}, dtype={t.tensors.dtype}, device={t.device}")
                print(
                    f"  mean={t.tensors.mean().item():.6f}, std={t.tensors.std().item():.6f}, min={t.tensors.min().item():.6f}, max={t.tensors.max().item():.6f}")

            # input_proj check (DQDETR has self.input_proj as ModuleList)
            print("\n=== INPUT_PROJ ===")
            for i, proj in enumerate(model.input_proj):
                conv = proj[0]  # assume seq: Conv2d, GroupNorm
                print(f"input_proj[{i}].conv.in_channels={conv.in_channels}, out_channels={conv.out_channels}")

            # build srcs as in DQDETR.forward
            srcs, masks = [], []
            for l, feat in enumerate(features):
                # feat may be NestedTensor
                if hasattr(feat, "decompose"):
                    src, mask = feat.decompose()
                elif hasattr(feat, "tensors"):
                    src, mask = feat.tensors, feat.mask
                else:
                    src, mask = feat, samples.mask
                print(f"srcs raw stage {l}: shape={src.shape}, mask shape={(mask.shape if mask is not None else None)}")
                # project
                p = model.input_proj[min(l, len(model.input_proj) - 1)]
                src_proj = p(src)
                print(
                    f"  -> after input_proj shape={src_proj.shape}, mean={src_proj.mean().item():.6f}, std={src_proj.std().item():.6f}")

                srcs.append(src_proj);
                masks.append(mask)

            # pooled statistics as transformer would see (adaptive avg pool)
            print("\n=== POOLED STATS (for dynamic-query-like checks) ===")
            for i, s in enumerate(srcs):
                pooled = F.adaptive_avg_pool2d(s, (1, 1)).flatten(1)  # B x C
                print(
                    f"pooled stage {i}: mean={pooled.mean().item():.6f}, std={pooled.std().item():.6f}, min={pooled.min().item():.6f}, max={pooled.max().item():.6f}")

            # check class head logits for one forward pass (use model up to outputs_class)
            try:
                out = model(samples)
                logits = out['pred_logits']  # shape: n_dec, bs, nq, num_classes  OR last: bs,nq,num_classes depending
                boxes = out['pred_boxes']
                print("\n=== MODEL OUTPUTS ===")
                if isinstance(logits, torch.Tensor):
                    print("pred_logits shape:", logits.shape, " mean:", logits.mean().item(), " std:",
                          logits.std().item())
                    # show softmax / sigmoid stats
                    if logits.dim() == 4:
                        last = logits[-1]  # take last layer
                    else:
                        last = logits
                    if last.shape[-1] > 1:
                        probs = last.softmax(-1)
                        print("  probs mean:", probs.mean().item(), " probs top1 mean:", probs.max(-1)[0].mean().item())
                    else:
                        probs = last.sigmoid()
                        print("  sigmoid mean:", probs.mean().item())
                else:
                    print("pred_logits not tensor:", type(logits))
                print("pred_boxes sample:", boxes.shape if isinstance(boxes, torch.Tensor) else None)
            except Exception as e:
                print("Model forward to outputs failed with:", e)

            # check optimizer params & whether classification head requires grad
            print("\n=== PARAMS / OPTIMIZER CHECK ===")
            # check class_embed requires_grad
            if hasattr(model, 'class_embed'):
                for i, ce in enumerate(model.class_embed):
                    print(
                        f"class_embed[{i}].weight.requires_grad = {ce.weight.requires_grad}, bias.requires_grad = {ce.bias.requires_grad}")
            # print some optimizer-like info if you have optimizer variable in scope (optional)
        model.train()

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    ccm_params = args.ccm_params

    _cnt = 0
    i = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        # i = i + 1
        # if i < 300:
        #     continue

        if isinstance(samples, list):
            samples = [sample.to(device) for sample in samples]
        else:
            samples = samples.to(device)
        # debug_once(model, samples, targets)

        ccm_targets = []
        for i in range(len(targets)):
            tgt_num = targets[i]['labels'].shape[0]
            t = 0
            for j in range(len(ccm_params)):
                if tgt_num >= ccm_params[j]:
                    t = j + 1
            ccm_targets.append(t)
     
        ccm_targets = torch.tensor(ccm_targets, dtype=torch.int64).to(device)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in t.items()}
            for t in targets
        ]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
        
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            ccm_loss = CCM_LOSS(outputs['pred_bbox_number'], ccm_targets)
            losses += ccm_coeff * ccm_loss

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}

        loss_dict_reduced_scaled['ccm_loss'] = ccm_loss
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})

    return resstat


def get_visualization_output_dir(args):
    """Return the configured visualization directory or the default output path."""
    vis_output_dir = getattr(args, 'vis_output_dir', '')
    if vis_output_dir:
        return vis_output_dir
    return os.path.join(getattr(args, 'output_dir', ''), 'visualizations')


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    if args.is_vis:
        all_predictions = {}
    visualization_output_dir = None
    if getattr(args, 'visualize', False) and utils.is_main_process():
        visualization_output_dir = get_visualization_output_dir(args)
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only
    # i = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        # i = i + 1
        # if i == 100:
        #     break
        if isinstance(samples, list):
            samples = [sample.to(device) for sample in samples]
        else:
            samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
                                      
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, outputs['num_select'])

        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        # res = {}
        # for target, output in zip(targets, results):
        #     h, w = target["size"]
        #     scale_fct = torch.tensor([w, h, w, h], device=output['boxes'].device)
        #
        #     # 将 target bbox 从归一化 xyxy -> xywh 像素
        #     target_boxes = target['boxes'] * scale_fct  # xyxy
        #     x_min, y_min, x_max, y_max = target_boxes.unbind(1)
        #     target['boxes'] = torch.stack([x_min, y_min, x_max - x_min, y_max - y_min], dim=1)  # xywh
        #
        #     # 将 output bbox 转为 xywh pixel（CocoEvaluator 内部也会做）
        #     output_boxes = output['boxes']  # 已经是像素 xyxy
        #     output['boxes'] = output_boxes  # 不用改，内部会 convert_to_xywh
        #
        #     # 构建 image_id -> prediction dict
        #     res[target['image_id'].item()] = output

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if args.is_vis:
            for target, output in zip(targets, results):
                img_id = target['image_id'].item()

                # 🔥 获取图片路径（关键）
                if "image_path" in target:
                    img_path = target["image_path"]
                elif "file_name" in target:
                    img_path = target["file_name"]
                else:
                    img_path = None

                all_predictions[img_id] = {
                    "boxes": output["boxes"].detach().cpu().numpy(),
                    "scores": output["scores"].detach().cpu().numpy(),
                    "labels": output["labels"].detach().cpu().numpy(),
                    "img_path": img_path,
                    "gt_boxes": target['boxes'].detach().cpu().numpy(),
                    "gt_labels": target['labels'].detach().cpu().numpy()
                }

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if visualization_output_dir is not None:
            for target, output in zip(targets, results):
                image_id = target['image_id'].item()
                image_path = target.get('image_path')
                try:
                    save_detection_visualization(
                        image_path=image_path,
                        image_id=image_id,
                        prediction=output,
                        gt_boxes=target['boxes'],
                        gt_labels=target['labels'],
                        save_dir=visualization_output_dir,
                        score_thresh=getattr(args, 'vis_score_thresh', 0.3),
                    )
                except Exception as exc:
                    print(
                        f"[WARN] Failed to save visualization for image_id {image_id} "
                        f"at path: {image_path}: {exc}"
                    )

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
        
        if args.save_results:
            # res_score = outputs['res_score']
            # res_label = outputs['res_label']
            # res_bbox = outputs['res_bbox']
            # res_idx = outputs['res_idx']

            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                
                # img_h, img_w = tgt['orig_size'].unbind()
                # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                # _res_bbox = res['boxes'] / scale_fct
                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                # import ipdb;ipdb.set_trace()

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

        if args.is_vis:
            if coco_evaluator is not None and 'bbox' in coco_evaluator.coco_eval:
                print("[INFO] Computing per-image AP...")

                img_ap = compute_per_image_ap(coco_evaluator.coco_eval['bbox'])
                vis_dir = os.path.join(output_dir, "topk_vis")

                visualize_topk_split_gt_pred(
                    data_loader.dataset,
                    all_predictions,
                    img_ap,
                    os.path.join(output_dir, "topk_vis"),
                    topk=5,
                    score_thresh=0.3
                )
        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]



    return stats, coco_evaluator

  
