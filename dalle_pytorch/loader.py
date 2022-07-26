from pathlib import Path
from random import randint, choice
import numpy as np

import PIL

from torch.utils.data import Dataset
from torchvision import transforms as T


class TextImageDataset(Dataset):
    def __init__(self,
                 folder,
                 text_len=256,
                 image_size=128,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 tokenizer=None,
                 negatives_path=None,
                 shuffle=False,
                 clip_embeddings=False
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.shuffle = shuffle
        self.clip_embeddings = clip_embeddings
        path = Path(folder)

        text_files = [*path.glob('**/*.txt')]
        image_files = [
            *path.glob('**/*.png'), *path.glob('**/*.jpg'),
            *path.glob('**/*.jpeg'), *path.glob('**/*.bmp')
        ]
        text_files = {text_file.stem: text_file for text_file in text_files}
        image_files = {image_file.stem: image_file for image_file in image_files}
        keys = ((image_files.keys() & text_files.keys()))
        self.text_files = {k: v for k, v in text_files.items() if k in keys}
        self.image_files = {k: v for k, v in image_files.items() if k in keys}
        
        if clip_embeddings:
            clip_files = [*path.glob('**/*.npy')]
            clip_files = {clip_file.stem: clip_file for clip_file in clip_files}
            keys = ((image_files.keys() & text_files.keys() & clip_files.keys()))
            self.clip_files = {k: v for k, v in clip_files.items() if k in keys}
        
        self.keys = list(keys)
        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
            if img.mode != 'RGB' else img),
            T.RandomResizedCrop(image_size,
                                scale=(self.resize_ratio, 1.),
                                ratio=(1., 1.)),
            T.ToTensor()
        ])
        self.negative_embeds = np.load(negatives_path) if negatives_path is not None else None

    def __len__(self):
        return len(self.keys)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        key = self.keys[ind]
        text_file = self.text_files[key]
        image_file = self.image_files[key]            

        descriptions = text_file.read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)
        try:
            image_tensor = self.image_transform(PIL.Image.open(image_file))
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)
        
        if self.clip_embeddings:
            clip_file = self.clip_files[key]
            try:
                clip_array = np.load(clip_file)
                clip_img = clip_array[:512]
                clip_txt = clip_array[512:]

            except:
                print(f"An exception occurred trying to load file {clip_file}.")
                print(f"Skipping index {ind}")
                return self.skip_sample(ind)
            
        if self.negative_embeds is not None:
            idx_file = int(str(text_file).split("/")[-1].split(".")[0])
            clip_embeds = self.negative_embeds[idx_file]
            past_clip_img = clip_embeds[0][:512]
            past_clip_txt = clip_embeds[0][512:]
            fut_clip = clip_embeds[1]
            neg_clips = clip_embeds[2:]

            # Success
            return tokenized_text, image_tensor, past_clip_img, past_clip_txt, fut_clip, neg_clips
        
        else:
            if self.clip_embeddings:
                return tokenized_text, image_tensor, clip_img, clip_txt
            else:
                return tokenized_text, image_tensor