import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from skimage import color  # require skimage
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class ColorizationDataset(BaseDataset):
  
    @staticmethod
    def modify_commandline_options(parser, is_train):
   
        parser.set_defaults(input_nc=1, output_nc=2, direction='AtoB')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir, opt.max_dataset_size))
        assert(opt.input_nc == 1 and opt.output_nc == 2 and opt.direction == 'AtoB')
        self.transform = get_transform(self.opt, convert=False)

    def __getitem__(self, index):

        path = self.AB_paths[index]
        im = Image.open(path).convert('RGB')
        im = self.transform(im)
        im = np.array(im)
        lab = color.rgb2lab(im).astype(np.float32)
        lab_t = transforms.ToTensor()(lab)
        A = lab_t[[0], ...] / 50.0 - 1.0
        B = lab_t[[1, 2], ...] / 110.0
        return {'A': A, 'B': B, 'A_paths': path, 'B_paths': path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
