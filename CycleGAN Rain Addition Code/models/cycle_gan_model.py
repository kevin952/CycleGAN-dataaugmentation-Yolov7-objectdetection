import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np


def gauss(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    if sigma <= 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
    s = sigma ** 2
    sum_val = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center

            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / 2 * s)
            sum_val += kernel[i, j]
    kernel = kernel / sum_val
    return kernel

class CycleGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_A', 'G_B', 'G']
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'real_BD']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        
        else:
            self.model_names = ['G_A', 'G_B']
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.gauss_1 = gauss(11, 0.1)
        self.gauss_2 = gauss(11, 0.3)
        self.gauss_3 = gauss(5, 1)
        self.gauss_conv1 = torch.FloatTensor(self.gauss_1).expand(3, 1, 11, 11)
        self.gauss_conv1 = torch.nn.Parameter(data=self.gauss_conv1, requires_grad=False).cuda()

        self.gauss_conv2 = torch.FloatTensor(self.gauss_2).expand(3, 1, 11, 11)
        self.gauss_conv2 = torch.nn.Parameter(data=self.gauss_conv2, requires_grad=False).cuda()

        self.gauss_conv3 = torch.FloatTensor(self.gauss_3).expand(3, 1, 5, 5)
        self.gauss_conv3 = torch.nn.Parameter(data=self.gauss_conv3, requires_grad=False).cuda()
        self.pad = torch.nn.ReflectionPad2d(5)

        if self.isTrain: 
                self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netD_B = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
                if opt.lambda_identity > 0.0: 
                    pass
                    # assert(opt.input_nc == opt.output_nc)
                self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
                self.fake_A_pool_2 = ImagePool(opt.pool_size)  
                self.fake_B_pool = ImagePool(opt.pool_size)  
                # define loss functions
                self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
                self.criterionCycle = torch.nn.L1Loss()
                self.criterionIdt = torch.nn.L1Loss()
                # Leung Add
                self.criterionAtt = torch.nn.MSELoss()
                self.criterionL1 = torch.nn.L1Loss()
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_G)
                self.optimizers.append(self.optimizer_D)
        def gaussianBlur(self, x, flag):
            if flag == 1:
                return torch.nn.functional.conv2d(x, self.gauss_conv1, groups=3)
            elif flag == 2:
                return torch.nn.functional.conv2d(x, self.gauss_conv2, groups=3)
            elif flag == 3:
                return torch.nn.functional.conv2d(x, self.gauss_conv3, groups=3)

        def set_input(self, input):
            AtoB = self.opt.direction == 'AtoB'
            self.real_A = input['A' if AtoB else 'B'].to(self.device)
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
            self.real_BD = input['BD'].to(self.device)
            self.image_paths = input['A_paths' if AtoB else 'B_paths']

        def forward(self):
            """Run forward pass; called by both functions <optimize_parameters> and <test>."""
            self.fake_B = self.netG_A(self.real_A)
            self.rec_A = self.netG_B(self.fake_B)   


            self.fake_A = self.netG_B(self.real_B)  # G_B(B)
            self.rec_B = self.netG_A(self.real_BD)  # G_A(G_B(B))
        def backward_D_basic(self, netD, real, fake):
            # Real
            pred_real = netD(real)
            loss_D_real = self.criterionGAN(pred_real, True)
            # Fake
            pred_fake = netD(fake.detach())
            loss_D_fake = self.criterionGAN(pred_fake, False)
            # Combined loss and calculate gradients
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            return loss_D

        def backward_D_A(self):
            """Calculate GAN loss for discriminator D_A"""
            fake_B = self.fake_B_pool.query(self.fake_B)
            self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

            self.backward_D_basic(self.netD_A, self.real_B, self.real_A)


        def backward_D_B(self):
            """Calculate GAN loss for discriminator D_B"""
            fake_A = self.fake_A_pool.query(self.fake_A)
            self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

            self.backward_D_basic(self.netD_B, self.real_A, self.real_B)

        def backward_A(self):
            self.loss_Att_A = self.criterionAtt(self.Att_A, torch.zeros_like(self.Att_A))

            self.loss_att = self.loss_Att_A
            self.loss_att.backward(retain_graph=True)

        def backward_G(self):
            """Calculate the loss for generators G_A and G_B"""
            # lambda_idt = self.opt.lambda_identity
            lambda_A = self.opt.lambda_A
            lambda_B = self.opt.lambda_B

            self.loss_change_FA = self.criterionL1(self.fake_A, self.real_BD) * 5.0
            # *****************

            # GAN loss D_A(G_A(A))
            self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True) * 2.0
            # GAN loss D_B(G_B(B))
            self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)


            # Forward cycle loss || G_B(G_A(A)) - A||
            self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * 10
            # Backward cycle loss || G_A(G_B(B)) - B||

            self.loss_cycle_B = self.criterionGAN(self.netD_A(self.rec_B), True) * 1

            self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_change_FA + \
                        self.loss_cycle_B +self.loss_change_FB_1
            # + self.loss_idt_B
            self.loss_G.backward()

        def optimize_parameters(self):
        
            # forward
            self.forward()      # compute fake images and reconstruction images.

            # Leung Add
            self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
       
            self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
            self.backward_G()             # calculate gradients for G_A and G_B
            self.optimizer_G.step()       # update G_A and G_B's weights
            # D_A and D_B
            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
            self.backward_D_A()      # calculate gradients for D_A
            self.backward_D_B()      # calculate graidents for D_B
            self.optimizer_D.step()  # update D_A and D_B's weights
