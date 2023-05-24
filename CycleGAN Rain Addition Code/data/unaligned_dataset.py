import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import cv2
import random
from RandAugment import RandAugment
from torchvision import transforms
import numpy as np


class UnalignedDataset(BaseDataset):

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        # Leung
        self.BD_paths = sorted(make_dataset(os.path.join(opt.dataroot, opt.phase + 'BD'), opt.max_dataset_size))
        # ***
        # Leung

        # ***
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image

        self.transform_A = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0, saturation=0, hue=0.3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        # self.transform_A.transforms.insert(0, RandAugment(2, 4))

        self.transform_B = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0, saturation=0, hue=0.3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])


    def __getitem__(self, index):

        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        BD_path = self.BD_paths[index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        BD_img = Image.open(BD_path).convert('RGB')
        # A_img = cv2.imread(A_path)
        # A_img = cv2.cvtColor(A_img, cv2.COLOR_BGR2HSV)
        # B_img = cv2.imread(B_path)
        # B_img = cv2.cvtColor(B_img, cv2.COLOR_BGR2HSV)
        # apply image transformation
        A = self.transform_A(A_img)

        seed = np.random.randint(2147483647)
        random.seed(seed)
        B = self.transform_B(B_img)
        random.seed(seed)
        BD = self.transform_B(BD_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'BD': BD}

    def __len__(self):

        return max(self.A_size, self.B_size)
