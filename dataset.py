import os
import random
import itertools

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler

import torchvision.transforms as T
from torchvision.io import read_image

from sklearn.model_selection import train_test_split


class LFWDataset(Dataset):
    def __init__(self, root_dir, faces, triplet_mode=False):
        self.root_dir = root_dir

        # [face_id, img_name]
        self.faces = faces

        # {face_id: [img_name1, img_name2, ...]}
        self.id2faces = {}
        for face_id, img_name in self.faces:
            if face_id in self.id2faces:
                self.id2faces[face_id].append(img_name)
            else:
                self.id2faces[face_id] = [img_name]

        # [face_id, (anchor, positive)]
        self.labled_pairs = []
        for face_id, img_names in self.id2faces.items():
            if len(img_names) > 1:
                for anchor, positive in itertools.permutations(img_names, 2):
                    self.labled_pairs.append((face_id, (anchor, positive)))
            else:
                self.labled_pairs.append((face_id, (img_names[0], None)))

        random.shuffle(self.labled_pairs)

        self.transform = T.Compose([T.ConvertImageDtype(torch.float32),
                                    T.Resize((224, 224)),
                                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])

        self.triplet_mode = triplet_mode

    def __len__(self):
        if self.triplet_mode:
            return len(self.labled_pairs)
        else:
            return len(self.faces)

    def __getitem__(self, idx):
        if self.triplet_mode:
            pos_id, (anchor, positive) = self.labled_pairs[idx]

            neg_id = random.choice([_id for _id in self.id2faces if _id != pos_id])
            negative = random.choice(self.id2faces[neg_id])

            # load images
            anchor_img = read_image(os.path.join(self.root_dir, str(pos_id), anchor))

            if positive is not None:
                pos_img = read_image(os.path.join(self.root_dir, str(pos_id), positive))
            else:
                pos_img = T.functional.hflip(anchor_img)

            neg_img = read_image(os.path.join(self.root_dir, str(neg_id), negative))

            # transform
            anchor_img, pos_img, neg_img = list(map(self.transform,
                                                    [anchor_img, pos_img, neg_img]))

            return anchor_img, pos_img, neg_img
        else:
            face_id, img_name = self.faces[idx]

            label = torch.tensor(face_id, dtype=torch.int64)

            img = read_image(os.path.join(self.root_dir, str(face_id), img_name))
            img = self.transform(img)

            return label, img

def load_data(root_dir):
    faces = []
    for face_id in os.listdir(root_dir):
        for img_name in os.listdir(os.path.join(root_dir, face_id)):
            faces.append((int(face_id), img_name))

    train_faces, test_faces = train_test_split(faces, test_size=0.2, random_state=42)

    return train_faces, test_faces

def create_dataloader(root_dir, faces, batch_size=1, training=False):
    dataset = LFWDataset(root_dir, faces, triplet_mode=True)

    n_samples = (10000 // batch_size + 1) * batch_size
    sampler = RandomSampler(dataset, num_samples=n_samples) if training else None
    dataloader = DataLoader(dataset, batch_size, sampler=sampler,
                            num_workers=2, drop_last=training)

    return dataloader
