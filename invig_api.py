import json
import os
import sys
from typing import List
from PIL import Image, ImageDraw
import numpy as np


def flatten_list(lists, flatten_ndarray=False):
    type_list = [tuple, list]
    if flatten_ndarray:
        type_list.append(np.ndarray)
    type_list = tuple(type_list)

    if not isinstance(lists, type_list):
        return lists

    all_data_list = []
    for l in lists:
        if isinstance(l, type_list):
            all_data_list.extend(flatten_list(l, flatten_ndarray=flatten_ndarray))
        else:
            all_data_list.append(l)

    return all_data_list

def read_jsonl(rpath: str):
    result = []
    with open(rpath, 'rt') as f:
        for line in f:
            result.append(json.loads(line.strip()))
    return result

def list_files(folders: List[str]) -> List[str]:
    files = []
    for folder in folders:
        if os.path.isdir(folder):
            files.extend([os.path.join(folder, d) for d in os.listdir(folder)])
        elif os.path.isfile(folder):
            files.append(folder)
        else:
            print('Path {} is invalid'.format(folder))
            sys.stdout.flush()
    return files

def list_all_files(dirs):
    sub_dirs = list_files(dirs)
    all_files = []
    all_dirs = []
    for d in sub_dirs:
        if os.path.isdir(d):
            all_dirs.append(d)
        else:
            all_files.append(d)
    if all_dirs:
        all_files.extend(list_all_files(all_dirs))
    return all_files

class INVIGAPI:
    def __init__(
            self,
            ann_root,
            img_root,
            split='train',
            version='21k'
    ):
        assert version in {'21k', '500k', '500+21k'}
        assert split in {'train', 'valid', 'test'}
        if version != '21k': assert split == 'train'

        self.split = split
        self.version = version
        self.ann_root = ann_root
        self.img_root = {
            '21k': os.path.join(img_root, 'invig21k_imgs'),
            '500k': os.path.join(img_root, 'invig500k_imgs'),
        }

        self.ann_files = list_all_files([ann_root])
        self.img_root_21k = os.path.join(img_root, 'invig21k_imgs')
        self.img_root_500k = os.path.join(img_root, 'invig500k_imgs')

        self.anns = self._init_anns()
        self.idx_map = self._init_idx_map()

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        return self.anns[idx]

    def _init_anns(self):

        def add_source(anns, source=self.version):
            for ann in anns:
                ann['source'] = source
            return anns

        self.anns = []
        if self.version == '21k':
            ann_file = os.path.join(self.ann_root, 'invig21k_anns', f"invig21k_{self.split}_anns.jsonl")
            self.anns.extend(add_source(read_jsonl(ann_file)))
        elif self.version == '500k':
            ann_file = os.path.join(self.ann_root, 'invig500k_anns', f"invig500k_anns.jsonl")
            self.anns.extend(add_source(read_jsonl(ann_file)))
        elif self.version == '500+21k':
            ann_file_21k = os.path.join(self.ann_root, 'invig21k_anns', f"invig21k_train_anns.jsonl")
            ann_file_500k = os.path.join(self.ann_root, 'invig500k_anns', "invig500k_anns.jsonl")
            self.anns.extend(add_source(read_jsonl(ann_file_21k), '21k'))
            self.anns.extend(add_source(read_jsonl(ann_file_500k), '500k'))

        return self.anns

    def _init_idx_map(self):
        self.old_idx_to_new = {}
        for i, ann in enumerate(self.anns):
            old_idx = f"{ann['source']}_{ann['id']}"
            self.old_idx_to_new[old_idx] = i
        return self.old_idx_to_new

    def get_image(self, idx):
        ann = self.anns[idx]
        img_file = os.path.join(self.img_root[ann['source']], ann['filename'])
        return Image.open(img_file).convert('RGB')

    def get_ann(self, idx):
        return self.anns[idx]['ann']

    def get_bboxes(self, idx):
        return np.array(self.anns[idx]['ann']['bboxes'])

    def get_ref_bboxes(self, idx):
        return np.array(self.anns[idx]['ann']['ref_bboxes'])

    def get_dialogues(self, idx):
        ref_exp = self.anns[idx]['ann']['ref_exp']
        questions = self.anns[idx]['ann']['questions']
        answers = self.anns[idx]['ann']['answers']
        return [ref_exp] + flatten_list(list(zip(questions, answers)))

    def _get_turn(self, turn, total):
        if turn is None:
            turn = np.random.randint(total + 1)
        elif turn == -1:
            turn = total
        assert 0 <= turn <= total
        return turn

    def get_mvqa(self, idx, turn=None):
        """
        turn:
            the idx of the turn that you want to use for training.
            Default: None, meaning a random turn will be applied.
        """
        data = {}

        ref_exp = self.anns[idx]['ann']['ref_exp']
        questions = self.anns[idx]['ann']['questions']
        answers = self.anns[idx]['ann']['answers']

        q_num = len(questions)
        turn = self._get_turn(turn, q_num - 1)

        data['image'] = self.get_image(idx)
        data['ref_bboxes'] = self.get_ref_bboxes(idx)

        data['ref_exp'] = ref_exp
        data['prev_questions'] = questions[:turn]
        data['prev_answers'] = answers[:turn]
        data['question'] = questions[turn]
        data['gt_answer'] = answers[turn]

        return data

    def get_mvqg(self, idx, turn=None):
        """
        turn:
            the idx of the turn that you want to use for training.
            Default: None, meaning a random turn will be applied.
        """
        data = {}

        ref_exp = self.anns[idx]['ann']['ref_exp']
        questions = self.anns[idx]['ann']['questions']
        answers = self.anns[idx]['ann']['answers']

        q_num = len(questions)
        turn = self._get_turn(turn, q_num)

        data['image'] = self.get_image(idx)
        data['bboxes'] = self.get_bboxes(idx)

        data['ref_exp'] = ref_exp
        data['prev_questions'] = questions[:turn]
        data['prev_answers'] = answers[:turn]
        if turn < q_num:
            data['gt_question'] = questions[turn]
        else:
            data['gt_question'] = ''

        return data

    def get_mvg(self, idx, turn=-1):
        """
        turn:
            the idx of the turn that you want to use for training.
            Default: -1, meaning the last turn
        """
        data = {}

        ref_exp = self.anns[idx]['ann']['ref_exp']
        questions = self.anns[idx]['ann']['questions']
        answers = self.anns[idx]['ann']['answers']

        q_num = len(questions)
        turn = self._get_turn(turn, q_num)

        data['image'] = self.get_image(idx)
        data['bboxes'] = self.get_bboxes(idx)

        data['ref_exp'] = ref_exp
        data['prev_questions'] = questions[:turn]
        data['prev_answers'] = answers[:turn]

        data['gt_ref_bboxes'] = self.get_ref_bboxes(idx)

        return data

    def save(self, idx, save_path):
        """
        save_path:
            a directory you want to save and visualize the annotation.
        """
        os.makedirs(save_path, exist_ok=True)
        image = self.get_image(idx)
        ref_bboxes = self.get_ref_bboxes(idx)
        image_show = self._draw_bboxes(image, ref_bboxes)
        image_show.save(os.path.join(save_path, f'{idx}.jpg'))

        dialog = self.get_dialogues(idx)
        json.dump(dialog, open(os.path.join(save_path, f'{idx}.json'), 'w'), indent=2)

    def _draw_bboxes(self, image, bboxes):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        assert isinstance(image, Image.Image)

        bboxes = np.array(bboxes)
        if bboxes.ndim == 1:
            bboxes = bboxes[None, ...]
        assert bboxes.shape[-1] == 4 and bboxes.ndim == 2

        w, h = image.width, image.height
        for b in bboxes:
            img_show = ImageDraw.Draw(image)
            img_show.rectangle((float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                               outline="green", width=int(0.01 * min(w, h)))
        return image
