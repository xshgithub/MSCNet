# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from PIL import Image

from datasets.coco import ConvertCocoPolysToMask, CocoDetection

if __name__ == "__main__":
    # for debug only
    import os, sys

    sys.path.append(os.path.dirname(sys.path[0]))

import json
from pathlib import Path
import random
import os

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

from datasets.data_util import preparing_dataset
import datasets.multi_transforms as T
from util.box_ops import box_cxcywh_to_xyxy, box_iou

__all__ = ['build']


class label2compat():
    def __init__(self) -> None:
        self.category_map_str = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
                                 "11": 11, "13": 12, "14": 13, "15": 14, "16": 15, "17": 16, "18": 17, "19": 18,
                                 "20": 19, "21": 20, "22": 21, "23": 22, "24": 23, "25": 24, "27": 25, "28": 26,
                                 "31": 27, "32": 28, "33": 29, "34": 30, "35": 31, "36": 32, "37": 33, "38": 34,
                                 "39": 35, "40": 36, "41": 37, "42": 38, "43": 39, "44": 40, "46": 41, "47": 42,
                                 "48": 43, "49": 44, "50": 45, "51": 46, "52": 47, "53": 48, "54": 49, "55": 50,
                                 "56": 51, "57": 52, "58": 53, "59": 54, "60": 55, "61": 56, "62": 57, "63": 58,
                                 "64": 59, "65": 60, "67": 61, "70": 62, "72": 63, "73": 64, "74": 65, "75": 66,
                                 "76": 67, "77": 68, "78": 69, "79": 70, "80": 71, "81": 72, "82": 73, "84": 74,
                                 "85": 75, "86": 76, "87": 77, "88": 78, "89": 79, "90": 80}
        self.category_map = {int(k): v for k, v in self.category_map_str.items()}

    def __call__(self, target, img=None):
        labels = target['labels']
        res = torch.zeros(labels.shape, dtype=labels.dtype)
        for idx, item in enumerate(labels):
            res[idx] = self.category_map[item.item()] - 1
        target['label_compat'] = res
        if img is not None:
            return target, img
        else:
            return target


class label_compat2onehot():
    def __init__(self, num_class=80, num_output_objs=1):
        self.num_class = num_class
        self.num_output_objs = num_output_objs
        if num_output_objs != 1:
            raise DeprecationWarning("num_output_objs!=1, which is only used for comparison")

    def __call__(self, target, img=None):
        labels = target['label_compat']
        place_dict = {k: 0 for k in range(self.num_class)}
        if self.num_output_objs == 1:
            res = torch.zeros(self.num_class)
            for i in labels:
                itm = i.item()
                res[itm] = 1.0
        else:
            # compat with baseline
            res = torch.zeros(self.num_class, self.num_output_objs)
            for i in labels:
                itm = i.item()
                res[itm][place_dict[itm]] = 1.0
                place_dict[itm] += 1
        target['label_compat_onehot'] = res
        if img is not None:
            return target, img
        else:
            return target


class box_label_catter():
    def __init__(self):
        pass

    def __call__(self, target, img=None):
        labels = target['label_compat']
        boxes = target['boxes']
        box_label = torch.cat((boxes, labels.unsqueeze(-1)), 1)
        target['box_label'] = box_label
        if img is not None:
            return target, img
        else:
            return target


class RandomSelectBoxlabels():
    def __init__(self, num_classes, leave_one_out=False, blank_prob=0.8,
                 prob_first_item=0.0,
                 prob_random_item=0.0,
                 prob_last_item=0.8,
                 prob_stop_sign=0.2
                 ) -> None:
        self.num_classes = num_classes
        self.leave_one_out = leave_one_out
        self.blank_prob = blank_prob

        self.set_state(prob_first_item, prob_random_item, prob_last_item, prob_stop_sign)

    def get_state(self):
        return [self.prob_first_item, self.prob_random_item, self.prob_last_item, self.prob_stop_sign]

    def set_state(self, prob_first_item, prob_random_item, prob_last_item, prob_stop_sign):
        sum_prob = prob_first_item + prob_random_item + prob_last_item + prob_stop_sign
        assert sum_prob - 1 < 1e-6, \
            f"Sum up all prob = {sum_prob}. prob_first_item:{prob_first_item}" \
            + f"prob_random_item:{prob_random_item}, prob_last_item:{prob_last_item}" \
            + f"prob_stop_sign:{prob_stop_sign}"

        self.prob_first_item = prob_first_item
        self.prob_random_item = prob_random_item
        self.prob_last_item = prob_last_item
        self.prob_stop_sign = prob_stop_sign

    def sample_for_pred_first_item(self, box_label: torch.FloatTensor):
        box_label_known = torch.Tensor(0, 5)
        box_label_unknown = box_label
        return box_label_known, box_label_unknown

    def sample_for_pred_random_item(self, box_label: torch.FloatTensor):
        n_select = int(random.random() * box_label.shape[0])
        box_label = box_label[torch.randperm(box_label.shape[0])]
        box_label_known = box_label[:n_select]
        box_label_unknown = box_label[n_select:]
        return box_label_known, box_label_unknown

    def sample_for_pred_last_item(self, box_label: torch.FloatTensor):
        box_label_perm = box_label[torch.randperm(box_label.shape[0])]
        known_label_list = []
        box_label_known = []
        box_label_unknown = []
        for item in box_label_perm:
            label_i = item[4].item()
            if label_i in known_label_list:
                box_label_known.append(item)
            else:
                # first item
                box_label_unknown.append(item)
                known_label_list.append(label_i)
        box_label_known = torch.stack(box_label_known) if len(box_label_known) > 0 else torch.Tensor(0, 5)
        box_label_unknown = torch.stack(box_label_unknown) if len(box_label_unknown) > 0 else torch.Tensor(0, 5)
        return box_label_known, box_label_unknown

    def sample_for_pred_stop_sign(self, box_label: torch.FloatTensor):
        box_label_unknown = torch.Tensor(0, 5)
        box_label_known = box_label
        return box_label_known, box_label_unknown

    def __call__(self, target, img=None):
        box_label = target['box_label']  # K, 5

        dice_number = random.random()

        if dice_number < self.prob_first_item:
            box_label_known, box_label_unknown = self.sample_for_pred_first_item(box_label)
        elif dice_number < self.prob_first_item + self.prob_random_item:
            box_label_known, box_label_unknown = self.sample_for_pred_random_item(box_label)
        elif dice_number < self.prob_first_item + self.prob_random_item + self.prob_last_item:
            box_label_known, box_label_unknown = self.sample_for_pred_last_item(box_label)
        else:
            box_label_known, box_label_unknown = self.sample_for_pred_stop_sign(box_label)

        target['label_onehot_known'] = label2onehot(box_label_known[:, -1], self.num_classes)
        target['label_onehot_unknown'] = label2onehot(box_label_unknown[:, -1], self.num_classes)
        target['box_label_known'] = box_label_known
        target['box_label_unknown'] = box_label_unknown

        return target, img


class RandomDrop():
    def __init__(self, p=0.2) -> None:
        self.p = p

    def __call__(self, target, img=None):
        known_box = target['box_label_known']
        num_known_box = known_box.size(0)
        idxs = torch.rand(num_known_box)
        # indices = torch.randperm(num_known_box)[:int((1-self).p*num_known_box + 0.5 + random.random())]
        target['box_label_known'] = known_box[idxs > self.p]
        return target, img


class BboxPertuber():
    def __init__(self, max_ratio=0.02, generate_samples=1000) -> None:
        self.max_ratio = max_ratio
        self.generate_samples = generate_samples
        self.samples = self.generate_pertube_samples()
        self.idx = 0

    def generate_pertube_samples(self):
        import torch
        samples = (torch.rand(self.generate_samples, 5) - 0.5) * 2 * self.max_ratio
        return samples

    def __call__(self, target, img):
        known_box = target['box_label_known']  # Tensor(K,5), K known bbox
        K = known_box.shape[0]
        known_box_pertube = torch.zeros(K, 6)  # 4:bbox, 1:prob, 1:label
        if K == 0:
            pass
        else:
            if self.idx + K > self.generate_samples:
                self.idx = 0
            delta = self.samples[self.idx: self.idx + K, :]
            known_box_pertube[:, :4] = known_box[:, :4] + delta[:, :4]
            iou = (torch.diag(
                box_iou(box_cxcywh_to_xyxy(known_box[:, :4]), box_cxcywh_to_xyxy(known_box_pertube[:, :4]))[0])) * (
                              1 + delta[:, -1])
            known_box_pertube[:, 4].copy_(iou)
            known_box_pertube[:, -1].copy_(known_box[:, -1])

        target['box_label_known_pertube'] = known_box_pertube
        return target, img


class RandomCutout():
    def __init__(self, factor=0.5) -> None:
        self.factor = factor

    def __call__(self, target, img=None):
        unknown_box = target['box_label_unknown']  # Ku, 5
        known_box = target['box_label_known_pertube']  # Kk, 6
        Ku = unknown_box.size(0)

        known_box_add = torch.zeros(Ku, 6)  # Ku, 6
        known_box_add[:, :5] = unknown_box
        known_box_add[:, 5].uniform_(0.5, 1)

        known_box_add[:, :2] += known_box_add[:, 2:4] * (torch.rand(Ku, 2) - 0.5) / 2
        known_box_add[:, 2:4] /= 2

        target['box_label_known_pertube'] = torch.cat((known_box, known_box_add))
        return target, img


class RandomSelectBoxes():
    def __init__(self, num_class=80) -> None:
        Warning("This is such a slow function and will be deprecated soon!!!")
        self.num_class = num_class

    def __call__(self, target, img=None):
        boxes = target['boxes']
        labels = target['label_compat']

        # transform to list of tensors
        boxs_list = [[] for i in range(self.num_class)]
        for idx, item in enumerate(boxes):
            label = labels[idx].item()
            boxs_list[label].append(item)
        boxs_list_tensor = [torch.stack(i) if len(i) > 0 else torch.Tensor(0, 4) for i in boxs_list]

        # random selection
        box_known = []
        box_unknown = []
        for idx, item in enumerate(boxs_list_tensor):
            ncnt = item.shape[0]
            nselect = int(random.random() * ncnt)  # close in both sides, much faster than random.randint

            item = item[torch.randperm(ncnt)]
            # random.shuffle(item)
            box_known.append(item[:nselect])
            box_unknown.append(item[nselect:])

        # box_known_tensor = [torch.stack(i) if len(i) > 0 else torch.Tensor(0,4) for i in box_known]
        # box_unknown_tensor = [torch.stack(i) if len(i) > 0 else torch.Tensor(0,4) for i in box_unknown]
        # print('box_unknown_tensor:', box_unknown_tensor)
        target['known_box'] = box_known
        target['unknown_box'] = box_unknown
        return target, img


def label2onehot(label, num_classes):
    """
    label: Tensor(K)
    """
    res = torch.zeros(num_classes)
    for i in label:
        itm = int(i.item())
        res[itm] = 1.0
    return res


class MaskCrop():
    def __init__(self) -> None:
        pass

    def __call__(self, target, img):
        known_box = target['known_box']
        h, w = img.shape[1:]  # h,w
        # imgsize = target['orig_size'] # h,w

        scale = torch.Tensor([w, h, w, h])

        # _cnt = 0
        for boxes in known_box:
            if boxes.shape[0] == 0:
                continue
            box_xyxy = box_cxcywh_to_xyxy(boxes) * scale
            for box in box_xyxy:
                x1, y1, x2, y2 = [int(i) for i in box.tolist()]
                img[:, y1:y2, x1:x2] = 0
                # _cnt += 1
        # print("_cnt:", _cnt)
        return target, img


dataset_hook_register = {
    'label2compat': label2compat,
    'label_compat2onehot': label_compat2onehot,
    'box_label_catter': box_label_catter,
    'RandomSelectBoxlabels': RandomSelectBoxlabels,
    'RandomSelectBoxes': RandomSelectBoxes,
    'MaskCrop': MaskCrop,
    'BboxPertuber': BboxPertuber,
}


def make_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # config the params for data aug
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]

    # update args from config files
    scales = getattr(args, 'data_aug_scales', scales)
    max_size = getattr(args, 'data_aug_max_size', max_size)
    scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
    scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)

    # resize them
    data_aug_scale_overlap = getattr(args, 'data_aug_scale_overlap', None)
    if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
        data_aug_scale_overlap = float(data_aug_scale_overlap)
        scales = [int(i * data_aug_scale_overlap) for i in scales]
        max_size = int(max_size * data_aug_scale_overlap)
        scales2_resize = [int(i * data_aug_scale_overlap) for i in scales2_resize]
        scales2_crop = [int(i * data_aug_scale_overlap) for i in scales2_crop]

    datadict_for_print = {
        'scales': scales,
        'max_size': max_size,
        'scales2_resize': scales2_resize,
        'scales2_crop': scales2_crop
    }
    print("data_aug_params:", json.dumps(datadict_for_print, indent=2))

    if image_set == 'train':
        if fix_size:
            return T.MultiModalWrapper(T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize([(max_size, max(scales))]),
                # T.RandomResize([(512, 512)]),
                normalize,
            ]))

        if strong_aug:
            import datasets.sltransform as SLT

            return T.MultiModalWrapper(T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        T.RandomResize(scales2_resize),
                        T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                SLT.RandomSelectMulti([
                    SLT.RandomCrop(),
                    SLT.LightingNoise(),
                    SLT.AdjustBrightness(2),
                    SLT.AdjustContrast(2),
                ]),
                normalize,
            ]))

        return T.MultiModalWrapper(T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales2_resize),
                    T.RandomSizeCrop(*scales2_crop),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ]))

    if image_set in ['val', 'eval_debug', 'train_reg', 'test']:

        if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == 'INFO':
            print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
            return T.MultiModalWrapper(T.Compose([
                T.ResizeDebug((1280, 800)),
                normalize,
            ]))

        return T.MultiModalWrapper(T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ]))

    raise ValueError(f'unknown {image_set}')


def get_aux_target_hacks_list(image_set, args):
    if args.modelname in ['q2bs_mask', 'q2bs']:
        aux_target_hacks_list = [
            label2compat(),
            label_compat2onehot(),
            RandomSelectBoxes(num_class=args.num_classes)
        ]
        if args.masked_data and image_set == 'train':
            # aux_target_hacks_list.append()
            aux_target_hacks_list.append(MaskCrop())
    elif args.modelname in ['q2bm_v2', 'q2bs_ce', 'q2op', 'q2ofocal', 'q2opclip', 'q2ocqonly']:
        aux_target_hacks_list = [
            label2compat(),
            label_compat2onehot(),
            box_label_catter(),
            RandomSelectBoxlabels(num_classes=args.num_classes,
                                  prob_first_item=args.prob_first_item,
                                  prob_random_item=args.prob_random_item,
                                  prob_last_item=args.prob_last_item,
                                  prob_stop_sign=args.prob_stop_sign,
                                  ),
            BboxPertuber(max_ratio=0.02, generate_samples=1000),
        ]
    elif args.modelname in ['q2omask', 'q2osa']:
        if args.coco_aug:
            aux_target_hacks_list = [
                label2compat(),
                label_compat2onehot(),
                box_label_catter(),
                RandomSelectBoxlabels(num_classes=args.num_classes,
                                      prob_first_item=args.prob_first_item,
                                      prob_random_item=args.prob_random_item,
                                      prob_last_item=args.prob_last_item,
                                      prob_stop_sign=args.prob_stop_sign,
                                      ),
                RandomDrop(p=0.2),
                BboxPertuber(max_ratio=0.02, generate_samples=1000),
                RandomCutout(factor=0.5)
            ]
        else:
            aux_target_hacks_list = [
                label2compat(),
                label_compat2onehot(),
                box_label_catter(),
                RandomSelectBoxlabels(num_classes=args.num_classes,
                                      prob_first_item=args.prob_first_item,
                                      prob_random_item=args.prob_random_item,
                                      prob_last_item=args.prob_last_item,
                                      prob_stop_sign=args.prob_stop_sign,
                                      ),
                BboxPertuber(max_ratio=0.02, generate_samples=1000),
            ]
    else:
        aux_target_hacks_list = None

    return aux_target_hacks_list


class MultiModalCocoDetection(CocoDetection):
    def __init__(self, img_folders, ann_file, transforms, return_masks, aux_target_hacks=None):
        """
        img_folders: list of 4 folders for 4 modalities
        """
        super().__init__(img_folders[0], ann_file, transforms, return_masks, aux_target_hacks)
        self.img_folders = img_folders


    def __getitem__(self, idx):
        # === 1) 读取 annotations ===
        img_info = self.coco.loadImgs(self.ids[idx])[0]
        file_name = img_info['file_name']

        # 读取第一模态
        base_img_path = os.path.join(self.img_folders[0], file_name)
        base_img = Image.open(base_img_path).convert("RGB")
        W, H = base_img.size

        # annotations
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=self.ids[idx]))

        # for ann in anns:
        #     x, y, w, h = ann['bbox']
        #     ann['bbox'] = [x / W, y / H, w / W, h / H]
        target = {'image_id': self.ids[idx], 'annotations': anns, 'image_path': base_img_path}

        # prepare target
        base_img, target = self.prepare(base_img, target)

        # === 2) 读取其它模态 ===
        multimodal_imgs = []
        for folder in self.img_folders:
            img_path = os.path.join(folder, file_name)
            im = Image.open(img_path).convert("RGB")
            multimodal_imgs.append(im)

        # === 3) 同步 transform ===
        if self._transforms is not None:
            # MultiModalWrapper 会返回 list[Tensor] 和 target
            multimodal_imgs, target = self._transforms(multimodal_imgs, target)

        # === 4) 检查每个模态 ===
        for i, im in enumerate(multimodal_imgs):
            if not torch.is_tensor(im):
                raise TypeError(f"multimodal_imgs[{i}] 不是 Tensor，请确认 ToTensor 已经被调用")

        # === 5) 不再在通道维拼接 ===
        # 直接返回 list[Tensor]，每个 Tensor [3,H,W]
        # h, w = target['size']
        # boxes = target['boxes'].clone()
        # boxes[:, 0::2] /= w
        # boxes[:, 1::2] /= h
        # target['boxes'] = boxes

        return multimodal_imgs, target


def build_multimodal(image_set, args):
    root = Path(args.coco_path)
    modalities = args.modalities
    # modalities = ['sla', 'ssh']
    img_folders = [root / mod / image_set for mod in modalities]
    ann_file = root / "annotations" / f"instances_{image_set}2017.json"

    strong_aug = getattr(args, 'strong_aug', False)
    fix_size = getattr(args, 'fix_size', False)
    transforms = make_coco_transforms(image_set, fix_size=fix_size, strong_aug=strong_aug, args=args)
    aux_target_hacks_list = get_aux_target_hacks_list(image_set, args)

    dataset = MultiModalCocoDetection(
        img_folders=img_folders,
        ann_file=ann_file,
        transforms=transforms,
        return_masks=getattr(args, 'masks', False),
        aux_target_hacks=aux_target_hacks_list
    )

    return dataset


if __name__ == "__main__":
    # Objects365 Val example
    dataset_o365 = CocoDetection(
        '/path/Objects365/train/',
        "/path/Objects365/slannos/anno_preprocess_train_v2.json",
        transforms=None,
        return_masks=False,
    )
    print('len(dataset_o365):', len(dataset_o365))
