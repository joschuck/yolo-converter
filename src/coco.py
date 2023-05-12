import copy
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from tqdm import tqdm


# https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
# converts 80-index (val2014) to 91-index (paper)
coco91_to_coco80_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, None, 24, 25, None,
         None, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, None, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
         51, 52, 53, 54, 55, 56, 57, 58, 59, None, 60, None, None, 61, None, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
         None, 73, 74, 75, 76, 77, 78, 79, None]


def convert_coco_json(
    input_dir: str,
    output_dir: str,
    task: str = "detection",
    cls91to80: bool = False,
    split: Optional[Tuple[float, float, float]] = None
):
    make_dirs(output_dir)
    split_annotations(input_dir, output_dir, split)

    for json_file in sorted(Path(input_dir).resolve().glob('*.json')):
        folder_name = Path(output_dir) / 'labels' / json_file.stem.split("_")[
            -1]  # .replace('instances_', '')  # folder name
        folder_name.mkdir(exist_ok=True)

        with open(json_file) as f:
            data = json.load(f)

        # Create image-annotations dict
        images = {f"{x['id']:g}": x for x in data['images']}
        img_to_anns = defaultdict(list)
        for ann in data['annotations']:
            img_to_anns[ann['image_id']].append(ann)

        # Write individual labels file
        for img_id, annotations in tqdm(img_to_anns.items(), desc=f'Annotations {json_file}'):

            img = images['%g' % img_id]
            height, width, file_name = img['height'], img['width'], img['file_name'].split('/')[-1]

            lines = []
            for ann in annotations:
                if ann['iscrowd']:
                    continue

                # class
                cls = coco91_to_coco80_class[ann['category_id'] - 1] if cls91to80 else ann['category_id'] - 1

                line = [cls]

                if task == "detection":
                    # The COCO box format is [top left x, top left y, width, height]
                    box = np.array(ann['bbox'], dtype=np.float64)
                    box[:2] += box[2:] / 2  # xy top-left corner to center
                    box[[0, 2]] /= width  # normalize x
                    box[[1, 3]] /= height  # normalize y
                    if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                        continue
                    line += box.tolist()

                elif task.lower() in ("segmentation", "panoptic"):
                    if len(ann['segmentation']) > 1:
                        segmentation = merge_multi_segment(ann['segmentation'])
                        segmentation = (np.concatenate(segmentation, axis=0) / np.array([width, height])).reshape(-1).tolist()
                    else:
                        segmentation = [j for i in ann['segmentation'] for j in i]  # all segments concatenated
                        segmentation = (np.array(segmentation).reshape(-1, 2) / np.array([width, height])).reshape(-1).tolist()
                    line += segmentation

                if task.lower() == "keypoints":
                    keyp = ann.get('keypoints')
                    if keyp is None:
                        continue

                    assert len(keyp) % 3 == 0

                    # normalize and append keypoints
                    for i in range(0, len(keyp), 3):
                        k_x, k_y, v = tuple(map(int, keyp[i:i + 3]))
                        line += [k_x / width, k_y / height, v]

                if line not in lines:
                    lines.append(line)

            # write individual label file
            with open((folder_name / file_name).with_suffix('.txt'), 'a') as file:
                for line in lines:
                    file.write(" ".join(f"{elem:g}" for elem in line).rstrip() + '\n')

        # Write individual labels file
        with open(Path(output_dir) / f"{json_file.stem}.txt", 'w') as file:
            for image in images.values():
                file.write(f"./images/{image['file_name']}\n")


def split_annotations(
    input_dir: str,
    output_dir: str,
    split: Optional[Tuple[float, float, float]] = None
):
    json_files = list(Path(input_dir).resolve().glob('*.json'))

    if len(json_files) > 1:
        print("Dataset already split")
        return

    json_file = list(Path(input_dir).resolve().glob('*.split'))[0]
    with open(json_file) as f:
        ds_orig = json.load(f)

    # Create image-annotations dict
    img_to_anns = defaultdict(list)
    for ann in ds_orig['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    # convert to absolute number of images
    if split is not None:
        split = tuple(map(lambda p: int(p * len(img_to_anns)), split))

    # create 3 datasets out of one
    datasets = {}
    for split_name in ("train", "val", "test"):
        datasets[split_name] = {
            "licenses": ds_orig["licenses"],
            "info": ds_orig["info"],
            "categories": ds_orig["categories"],
            "images": [],
            "annotations": []
        }

    # Write individual labels file
    for index, (img_id, annotations) in tqdm(enumerate(img_to_anns.items()), desc=f'Splitting {json_file}'):
        split_name = "train"
        if split[0] <= index < sum(split[:2]):
            split_name = "val"
        elif sum(split[:2]) <= index:
            split_name = "test"

        # write annotation to correct split
        datasets[split_name]["annotations"].extend(annotations)

        # move image
        image_meta = next(image for image in ds_orig['images'] if image["id"] == img_id)
        folder_name = Path(output_dir) / 'images' / split_name
        folder_name.mkdir(exist_ok=True)

        old_path = Path(output_dir) / 'images' / image_meta['file_name']
        new_path = Path(output_dir) / 'images' / split_name / image_meta['file_name']
        shutil.move(old_path, new_path)

        # change path
        image_meta["file_name"] = f"{split_name}/{image_meta['file_name']}"
        datasets[split_name]["images"].append(image_meta)

    # write split ds
    for split_name, ds in datasets.items():
        with (Path(input_dir) / f"{json_file.stem}_{split_name}.json").open("w") as outfile:
            json.dump(ds, outfile)

def min_index(arr1, arr2):
    """Find a pair of indexes with the shortest distance.
    Args:
        arr1: (N, 2).
        arr2: (M, 2).
    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """Merge multi segments to one list.
    Find the coordinates with min distance between each segment,
    then connect these coordinates with one thin line to merge all
    segments into one.
    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...],
            each segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0]:idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s



def make_dirs(_dir: str):
    """
    Creates a clean directory for conversion output. Delete content if output dir already exists.
    """
    _dir = Path(_dir)

    labels_dir = _dir / 'labels'
    image_dir = _dir / 'images'
    if labels_dir.exists():
        shutil.rmtree(labels_dir)

    for p in labels_dir, image_dir:
        p.mkdir(parents=True, exist_ok=True)
