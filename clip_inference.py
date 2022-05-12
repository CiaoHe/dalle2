import shutil
import fsspec, math, json
from io import BytesIO
import time
import argparse
import os

import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm_data.load_clip import load_clip
from mm_data import get_dataset

###### Reader ######

def dataset_to_dataloader(dataset, batch_size, num_prepro_workers, input_format):
    """Create a pytorch dataloader from a dataset"""

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return default_collate(batch)
    
    data = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_prepro_workers,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_fn if input_format == "files" else None,
    )
    return data

class Reader:
    def __init__(self, input_dataset, batch_size, num_prepro_workers, input_format):
        self.dataloader = dataset_to_dataloader(input_dataset, batch_size, num_prepro_workers, 'files')
    
    def __iter__(self):
        for batch in self.dataloader:
            yield dict(image_tensor=batch[0], text_tensor=batch[1])

###### Mapper ######

class Mapper:
    """transforms images and text to into clip embeddings"""
    def __init__(self, enable_image, enable_text, clip_model:str, use_jit):
        self.enable_image = enable_image
        self.enable_text = enable_text
        model, _ = load_clip(clip_model, use_jit)
        self.img_encode = model.encode_image
        self.text_encode = model.encode_text
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __call__(self, item):
        with torch.no_grad():
            image_embs = None
            text_embs =None
            if self.enable_image:
                image_features = self.img_encode(item["image_tensor"].to(self.device))
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_embs = image_features.cpu().numpy()
            if self.enable_text:
                text_features = self.text_encode(item["text_tensor"].to(self.device))
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_embs = text_features.cpu().numpy()
            return {"image_embs": image_embs, "text_embs": text_embs}

###### Writer ######        

class OutputSink:
    """This output sink can save image, text embeddings as npy and metadata as parquet"""
    
    def __init__(self, output_folder, enable_text, enable_image, output_partition_count):
        self.enable_text = enable_text
        self.enable_image = enable_image
        self.fs, output_folder = fsspec.core.url_to_fs(output_folder)
        self.output_folder = output_folder
        self.img_emb_folder = output_folder + "/img_emb"
        self.text_emb_folder = output_folder + "/text_emb"
        self.batch_num = 0
        self.oom_partition_count = int(math.log10(output_partition_count)) + 1
        
        if enable_image:
            self.fs.makedirs(self.img_emb_folder, exist_ok=True)

        if enable_text:
            self.fs.makedirs(self.text_emb_folder, exist_ok=True)
    
        self.back_count = 0
        self.__init_batch()
        
    def __init_batch(self):
        self.image_embeddings = []
        self.text_embeddings = []
        self.batch_count = 0
    
    def add(self, sample):
        """
        add to buffers the image embeddings, text embeddings
        Parameters: sample:dict(image_embds, text_embeds)
        """

        self.batch_count += sample["image_embs"].shape[0] if self.enable_image else sample["text_embs"].shape[0]
        if self.enable_image:
            self.image_embeddings.append(sample["image_embs"])
        if self.enable_text:
            self.text_embeddings.append(sample["text_embs"])
    
    def __write_batch(self):
        """
        write a batch of embeddings and meta to npy and parquet
        """
        import numpy as np  # pylint: disable=import-outside-toplevel
        
        batch_num_str = str(self.batch_num).zfill(self.oom_partition_count)
        if self.enable_image:
            img_emb_mat = np.concatenate(self.image_embeddings)
            output_path_img = self.img_emb_folder + "/img_emb_" + batch_num_str
            with self.fs.open(output_path_img + ".npy", "wb") as f:
                npb = BytesIO()
                np.save(npb, img_emb_mat)
                f.write(npb.getbuffer())

        if self.enable_text:
            text_emb_mat = np.concatenate(self.text_embeddings)
            output_path_text = self.text_emb_folder + "/text_emb_" + batch_num_str

            with self.fs.open(output_path_text + ".npy", "wb") as f:
                npb = BytesIO()
                np.save(npb, text_emb_mat)
                f.write(npb.getbuffer())
        
        self.batch_num += 1

    def flush(self):
        if self.batch_count == 0:
            return
        self.__write_batch()
        
class NumpyWriter:
    """the numpy writer writes embeddings to folders img_emb, text_emb, and metadata"""

    def __init__(self, output_folder, enable_text, enable_image, output_partition_count):
        self.sink = OutputSink(
            output_folder, enable_text, enable_image, output_partition_count
        )

    def __call__(self, batch):
        self.sink.add(batch)

    def flush(self):
        self.sink.flush()

###### Runner ######
            
class Runner:
    def __init__(self, reader_builder, mapper_builder, writer_builder, output_partition_count):
        self.reader_builder = reader_builder
        self.mapper_builder = mapper_builder
        self.writer_builder = writer_builder
        self.output_partition_count = output_partition_count
        self.pbar = tqdm(total=output_partition_count)
        
    def __call__(self):
        reader = self.reader_builder()
        mapper = self.mapper_builder()
        writer = self.writer_builder()
        iterator = reader.__iter__()
        while True:
            start_time = time.perf_counter()
            try:
                batch = iterator.__next__()
            except StopIteration:
                break
            read_duration = time.perf_counter() - start_time
            start_time = time.perf_counter()
            # mapper embeddings
            embeddings = mapper(batch)
            inference_duration = time.perf_counter() - start_time
            start_time = time.perf_counter()
            writer(embeddings)
            write_duration = time.perf_counter() - start_time
            self.pbar.update(1)
            print("read: {:.2f}s, inference: {:.2f}s, write: {:.2f}s".format(read_duration, inference_duration, write_duration))
            writer.flush()
            
def main(
    input_dataset,
    mode,                           # train, val
    output_folder,
    enable_text = True,
    enable_image = True,
    clip_model="ViT-B/32",
    use_jit=True,
    write_batch_size = 10**6,
    num_prepro_workers = 8,
    input_format="files",
):
    if isinstance(input_dataset, str):
        assert input_dataset == 'CC_3M', f"input_dataset must be CC_3M, but got {input_dataset}"
        input_dataset, _ = get_dataset(input_dataset, split=mode, preprocess_clip_way=True)
    
    if not os.path.exists(output_folder):
        print("creating output folder:", output_folder)
        os.makedirs(output_folder)
    elif os.path.exists(output_folder) and os.listdir(output_folder):
        shutil.rmtree(output_folder)
    
    sample_count = len(input_dataset)
    write_batch_size = write_batch_size
    output_partition_count = int(sample_count / write_batch_size) + 1
    
    def reader_builder():
        return Reader(input_dataset, write_batch_size, num_prepro_workers, input_format)

    def mapper_builder():
        return Mapper(enable_image, enable_text, clip_model, use_jit)
    
    def writer_builder():
        return NumpyWriter(
            output_folder=output_folder,
            enable_text=enable_text,
            enable_image=enable_image,
            output_partition_count=output_partition_count,
        )
    
    runner = Runner(
        reader_builder=reader_builder,
        mapper_builder=mapper_builder,
        writer_builder=writer_builder,
        output_partition_count=output_partition_count,
    )
    
    runner()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", type=str, default="CC_3M")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--output_folder", type=str, default="output", required=True)
    parser.add_argument("--enable_text", type=bool, default=True)
    parser.add_argument("--enable_image", type=bool, default=True)
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--use_jit", type=bool, default=True)
    parser.add_argument("--write_batch_size", type=int, default=10**6)
    parser.add_argument("--num_prepro_workers", type=int, default=8)
    parser.add_argument("--input_format", type=str, default="files")
    
    args = parser.parse_args()
    main(
        input_dataset=args.input_dataset,
        mode=args.mode,
        output_folder=args.output_folder,
        enable_text=args.enable_text,
        enable_image=args.enable_image,
        clip_model=args.clip_model,
        use_jit=args.use_jit,
        write_batch_size=args.write_batch_size,
        num_prepro_workers=args.num_prepro_workers,
        input_format=args.input_format,
    )