import os
import json
import cv2
import base64
from matplotlib import image

from resize_right import resize

import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
from mm_data.tokenizer import tokenizer

import os
import os.path as op
from idea import MODULES

from idea.dataset.tsv import generate_lineidx, FileProgressingbar
from idea.dataset.transform import Compose
        


def img_from_base64(imagestring):
    jpgbytestring = base64.b64decode(imagestring)
    nparr = np.frombuffer(jpgbytestring, np.uint8)
    try:
        r = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return r
    except:
        return None

def resize_image_to(image, target_image_size):
    orig_h, orig_w = image.shape[-2:]
    
    if orig_h == target_image_size and orig_w == target_image_size:
        return image
    
    scale_factors = (target_image_size / orig_h, target_image_size / orig_w)
    
    return resize(image, scale_factors = scale_factors)

def l2norm(t):
    return F.normalize(t, p=2, dim=-1)

def normalize_img(img):
    return (img - 0.5) * 2

def unnomalize_img(img):
    return img / 2 + 0.5


class TSVFile(object):
    def __init__(self, tsv_file, lineidx_file=None, silence=True):
        self.tsv_file = tsv_file
        if lineidx_file is None:
            self.lineidx = op.splitext(tsv_file)[0] + '.lineidx'
        else:
            self.lineidx = lineidx_file
        self._fp = None
        self._lineidx = None
        self.silence = silence

        self._ensure_lineidx_loaded()

    def num_rows(self):
        return len(self._lineidx)

    def seek(self, idx):
        self._ensure_tsv_opened()
        pos = self._lineidx[idx]
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split('\t')]

    def seek_list(self, idxs, q):
        assert isinstance(idxs, list)
        self._ensure_tsv_opened()
        for idx in idxs:
            pos = self._lineidx[idx]
            self._fp.seek(pos)
            q.put([s.strip() for s in self._fp.readline().split('\t')])

    def close(self):
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def _ensure_lineidx_loaded(self):
        if not op.isfile(self.lineidx) and not op.islink(self.lineidx):
            generate_lineidx(self.tsv_file, self.lineidx)

        if self._lineidx is None:
            with open(self.lineidx, 'r') as fp:
                if not self.silence:
                    bar = FileProgressingbar(fp, "Loading lineidx {0}: ".format(self.lineidx))
                self._lineidx = []
                fpos = 0
                fsize = os.fstat(fp.fileno()).st_size
                while fpos != fsize:
                    i = fp.readline()
                    fpos = fp.tell()
                    self._lineidx.append(int(i.strip()))
                    if not self.silence:
                        bar.update()

    def _ensure_tsv_opened(self):
        if self._fp is None:
            self._fp = open(self.tsv_file, 'r')



@MODULES.register
class MMTSVDataset(Dataset):
    '''
        TSV dataset for tsv file
    '''    
    def __init__(self, tsv_file, lineidx_file=None, repeat_time=1, image_transforms=None, label_transforms=None, map_color=False, **kwargs):
        self.tsv = TSVFile(tsv_file, lineidx_file)
        self.repeat_time = repeat_time
        self.map_color = map_color  # whether to do BGR-> RGB
        self.infokey = kwargs.get('key', 'text')

        def build_transforms(transforms):
            if transforms is None: 
                return False
            else:
                if isinstance(transforms, list):
                    return Compose(transforms)
                else:
                    return transforms
        self.image_transforms = build_transforms(image_transforms)
        self.label_transforms = build_transforms(label_transforms)


    @property
    def real_len(self):
        return self.tsv.num_rows()

    def check(self, row):
        return True

    def read_img(self, row):
        try:
            img = img_from_base64(row[-1])
            if self.map_color:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            return img
        except Exception as e:
            return False

    def read_label(self, row):
        info = json.loads(row[1])
        # TODO(jiananw): Ugly getaround for CC 12M.
        # None value makes dataloader unhappy.
        if 'error_message' in info.keys():
            del info['error_message']
        return info

    def __getitem__(self, index):
        index = index % self.real_len
        row = self.tsv.seek(index)
        img = self.read_img(row)
        # TODO(jiananw): Relentlessly try to get a valid image. This destroys uniform sampling.
        # Temporary to get around data quality issue.
        while img == False:
            print("error img: {}".format(index))
            index += 1
            index = index % self.real_len
            row = self.tsv.seek(index)
            img = self.read_img(row)

        if self.image_transforms:
            img = self.image_transforms(img)

        label_info = self.read_label(row)[self.infokey]
        if self.label_transforms:
            label_info = self.label_transforms(label_info)

        return img, label_info

    def __len__(self):
        return int(self.tsv.num_rows() * self.repeat_time)


@MODULES.register
class MMTSVEmbed_Dataset(MMTSVDataset):
    """MMTSV CLIP embed dataset"""
    def __init__(self, clip_name:str='ViT-B/32', **kwargs):
        import clip 
        openai_clip, preprocess = clip.load(clip_name, 'cpu')
        self.clip = openai_clip
        self.preprocess = preprocess
        self.tokenizer_func = clip.tokenize
        super().__init__(**kwargs)
        
    @property
    def image_size(self):
        return self.clip.visual.input_resolution
    
    def __getitem__(self, index):
        index = index % self.real_len
        row = self.tsv.seek(index)
        img = self.read_img(row)

        while img == False:
            print("error img: {}".format(index))
            index += 1
            index = index % self.real_len
            row = self.tsv.seek(index)
            img = self.read_img(row)

        if self.preprocess:
            img = self.preprocess(img)
        elif self.image_transforms:
            img = self.image_transforms(img)

        text = self.read_label(row)[self.infokey]
        text = self.tokenizer_func(text)
        
        with torch.no_grad():
            # embed text
            text_embed = self.clip.encode_text(text).squeeze(0)
            # embed image
            image_embed = self.clip.encode_image(image.unsqueeze(0)).squeeze(0)
    
        return l2norm(image_embed).float(), l2norm(text_embed).float()