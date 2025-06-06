from __future__ import print_function
from random import seed
from logging import Formatter, StreamHandler, getLogger, FileHandler
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch.backends.cudnn
import numpy as np

from models import *
from utils import *
from miou import *
from loss_functions import *

from dataloader.Cityscapes import decode_labels
from dataset import get_dataset
from prettytable import PrettyTable


def set_converts(datasets, task):
    '''
    train_converts: converts for image-to-image translation training
    test_converts: converts for domain adaptation test
    tensorboard_converts: converts for tensorboard

    Examples

    1. MNIST <-> MNIST-M
        train_converts = ['M2MM', 'MM2M']
        test_converts = ['M2MM', 'MM2M']
        tensorboard_converts = ['M2MM', 'MM2M']
    2. MNIST <-> MNIST-M <-> USPS
        train_converts = ['M2MM', 'MM2M', 'M2U', 'U2M']
        test_converts = ['M2MM', 'MM2M', 'M2U', 'U2M', 'MM2U', 'U2MM']
        tensorboard_converts = ['M2MM', 'MM2M', 'M2U', 'U2M', 'MM2U', 'U2MM']
    3. GTA5 <-> Cityscapes
        train_converts = ['G2C', 'C2G']
        test_converts = ['G2C']
        tensorboard_converts = ['G2C', 'C2G']
    '''
    training_converts, test_converts = [], []
    center_dset = datasets[0]
    for source in datasets:  # source
        if not center_dset == source:
            training_converts.append(center_dset + '2' + source)
            training_converts.append(source + '2' + center_dset)
        if task == 'clf':
            for target in datasets:  # target
                if not source == target:
                    test_converts.append(source + '2' + target)
    if task == 'clf':
        tensorboard_converts = test_converts
    elif task == 'seg':
        test_converts.append('G2C')
        tensorboard_converts = training_converts
    else:
        raise Exeception("Does not support the task")

    return training_converts, test_converts, tensorboard_converts

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class Trainer:
    def __init__(self, args):
        self.args = args
        self.training_converts, self.test_converts, self.tensorboard_converts = set_converts(args.datasets, args.task)
        self.arg_use_entropy = False
        self.arg_use_vat = False
        self.arg_use_D_optimization = True  # 加入 p(y|x) 的优化
        self.arg_D_try1 = False
        self.arg_D_try2 = False
        self.arg_D_try3 = False
        self.arg_D_try4 = True
        self.grl_lambda = 1.0
        self.current_iter = 0
        self.total_iters = 10000
        self.rand_low_dim = 64

        if self.arg_D_try3:
            self.arg_D_try2 = True

        if self.arg_use_D_optimization:

            # 固定设置（根据 Encoder 结构）
            feature_channels = 64  # Encoder 最后一层输出通道数
            num_classes = 10       # 根据任务调整（如 MNIST=10）

            # 新增适配层 (Adapter)
            if self.arg_D_try1:
                self.adapter = nn.ModuleDict({
                    'M': nn.Sequential(
                        nn.Conv2d(feature_channels * num_classes, 3, kernel_size=1),
                        nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)
                    ),
                    'MM': nn.Sequential(
                        nn.Conv2d(feature_channels * num_classes, 3, kernel_size=1),
                        nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)
                    )
                })
            elif self.arg_D_try2:
                self.adapter = nn.ModuleDict({
                    'M': nn.Conv2d(64, 3, kernel_size=1),
                    'MM': nn.Conv2d(64, 3, kernel_size=1)
                })
            elif self.arg_D_try4:
                # self.adapter = nn.ModuleDict({
                #     'M': nn.Sequential(
                #         nn.Linear(self.rand_low_dim, 128 * 8 * 8),         # [B, 8192]
                #         nn.ReLU(),
                #         View([-1, 128, 8, 8]),                              # reshape to [B, 128, 8, 8]
                #         nn.Conv2d(128, 64, kernel_size=3, padding=1),
                #         nn.ReLU(),
                #         nn.Conv2d(64, 3, kernel_size=3, padding=1),
                #         nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)
                #     ),
                #     'MM': nn.Sequential(
                #         nn.Linear(self.rand_low_dim, 128 * 8 * 8),
                #         nn.ReLU(),
                #         View([-1, 128, 8, 8]),
                #         nn.Conv2d(128, 64, kernel_size=3, padding=1),
                #         nn.ReLU(),
                #         nn.Conv2d(64, 3, kernel_size=3, padding=1),
                #         nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)
                #     )
                # })
                self.adapter = nn.ModuleDict({
                    'M': nn.Conv2d(64, 3, kernel_size=1),
                    'MM': nn.Conv2d(64, 3, kernel_size=1)
                })
                self.rand_proj = nn.ModuleDict({
                    'M': nn.Linear(feature_channels * num_classes, self.rand_low_dim, bias=False),
                    'MM': nn.Linear(feature_channels * num_classes, self.rand_low_dim, bias=False)
                })
            # 新增 p_projector、gate
            self.p_projector = nn.ModuleDict({
                dset: nn.Linear(num_classes, feature_channels)  # e.g., 10 -> 64
                for dset in self.args.datasets
            })
            self.gate = nn.ModuleDict({
                dset: nn.Conv2d(feature_channels, 1, kernel_size=1)
                for dset in self.args.datasets
            })
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.adapter = self.adapter.to(self.device)
            self.p_projector = self.p_projector.to(self.device)
            self.gate = self.gate.to(self.device)
            self.rand_proj = self.rand_proj.to(self.device)

        if args.task == 'seg':
            self.imsize = (2 * args.imsize, args.imsize)  # (width, height)
            self.best_miou = 0.
        elif args.task == 'clf':
            self.imsize = (args.imsize, args.imsize)
            self.acc = dict()
            self.best_acc = dict()
            for cv in self.test_converts:
                self.best_acc[cv] = 0.

        # data loader
        self.train_loader, self.test_loader, self.data_iter = dict(), dict(), dict()
        for dset in self.args.datasets:
            train_loader, test_loader = get_dataset(dataset=dset, batch=self.args.batch,
                                                    imsize=self.imsize, workers=self.args.workers)
            self.train_loader[dset] = train_loader
            self.test_loader[dset] = test_loader

        self.nets, self.optims, self.losses = dict(), dict(), dict()
        self.loss_fns = Loss_Functions(args)

        self.writer = SummaryWriter('./tensorboard/%s' % args.ex)
        self.logger = getLogger()
        self.checkpoint = './checkpoint/%s/%s' % (args.task, args.ex)
        self.step = 0

        self.name_classes_19 = [
            "road", "sidewalk", "building", "wall", "fence", "pole", "trafflight", "traffsign", "vegetation",
            "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
        ]
        

    def set_default(self):
        torch.backends.cudnn.benchmark = True

        ## Random Seed ##
        print("Random Seed: ", self.args.manualSeed)
        seed(self.args.manualSeed)
        torch.manual_seed(self.args.manualSeed)
        torch.cuda.manual_seed_all(self.args.manualSeed)

        ## Logger ##
        file_log_handler = FileHandler(self.args.logfile)
        self.logger.addHandler(file_log_handler)
        stderr_log_handler = StreamHandler(sys.stdout)
        self.logger.addHandler(stderr_log_handler)
        self.logger.setLevel('INFO')
        formatter = Formatter()
        file_log_handler.setFormatter(formatter)
        stderr_log_handler.setFormatter(formatter)

    def save_networks(self):
        if not os.path.exists(self.checkpoint+'/%d' % self.step):
            os.mkdir(self.checkpoint+'/%d' % self.step)
        for key in self.nets.keys():
            if key == 'D':
                for dset in self.args.datasets:
                    torch.save(self.nets[key][dset].state_dict(),
                               self.checkpoint + '/%d/net%s_%s.pth' % (self.step, key, dset))
            elif key == 'T':
                for cv in self.test_converts:
                    torch.save(self.nets[key][cv].state_dict(),
                                self.checkpoint + '/%d/net%s_%s.pth' % (self.step, key, cv))
            else:
                torch.save(self.nets[key].state_dict(), self.checkpoint + '/%d/net%s.pth' % (self.step, key))

    def load_networks(self, step):
        self.step = step
        for key in self.nets.keys():
            if key == 'D':
                for dset in self.args.datasets:
                    self.nets[key][dset].load_state_dict(torch.load(self.checkpoint
                                                                    + '/%d/net%s_%s.pth' % (step, key, dset)))
            elif key == 'T':
                if self.args.task == 'clf':
                    for cv in self.test_converts:
                        self.nets[key][cv].load_state_dict(torch.load(self.checkpoint
                                                                      + '/%d/net%s_%s.pth' % (step, key, cv)))
            else:
                self.nets[key].load_state_dict(torch.load(self.checkpoint + '/%d/net%s.pth' % (step, key)))

    def set_networks(self):
        self.nets['E'] = Encoder()
        self.nets['G'] = Generator()
        self.nets['S'] = Separator(self.imsize, self.training_converts)
        self.nets['D'] = dict()
        for dset in self.args.datasets:
            if self.args.task == 'clf':
                if dset == 'U':
                    self.nets['D'][dset] = Discriminator_USPS()
                else:
                    self.nets['D'][dset] = Discriminator_MNIST()
            else:
                self.nets['D'][dset] = PatchGAN_Discriminator()
        self.nets['T'] = dict()
        for cv in self.test_converts:  
            if self.args.task == 'clf':
                self.nets['T'][cv] = Classifier()
            elif self.args.task == 'seg':
                self.nets['T'][cv] = drn26()
                # self.nets['T'][cv] = GTA5_drn26()  # You may use your own pretrained drn26 on GTA5

        # initialization
        for net in self.nets.keys():
            if net == 'D':
                for dset in self.args.datasets:
                    init_params(self.nets[net][dset])
            elif net == 'T':
                if self.args.task == 'clf':
                    for cv in self.test_converts:
                        init_params(self.nets[net][cv])
                elif self.args.task == 'seg':
                    pass
            else:
                init_params(self.nets[net])
        self.nets['P'] = VGG19()

        for net in self.nets.keys():
            if net == 'D':
                for dset in self.args.datasets:
                    self.nets[net][dset].cuda()
            elif net == 'T':
                for cv in self.test_converts:
                    self.nets[net][cv].cuda()
            else:
                self.nets[net].cuda()

    def set_optimizers(self):
        self.optims['E'] = optim.Adam(self.nets['E'].parameters(), lr=self.args.lr_dra,
                                      betas=(self.args.beta1, 0.999),
                                      weight_decay=self.args.weight_decay_dra)

        self.optims['D'] = dict()
        for dset in self.args.datasets:
            self.optims['D'][dset] = optim.Adam(self.nets['D'][dset].parameters(), lr=self.args.lr_dra,
                                                betas=(self.args.beta1, 0.999),
                                                weight_decay=self.args.weight_decay_dra)

        self.optims['G'] = optim.Adam(self.nets['G'].parameters(), lr=self.args.lr_dra,
                                      betas=(self.args.beta1, 0.999),
                                      weight_decay=self.args.weight_decay_dra)

        self.optims['S'] = optim.Adam(self.nets['S'].parameters(), lr=self.args.lr_dra,
                                      betas=(self.args.beta1, 0.999),
                                      weight_decay=self.args.weight_decay_dra)
        
        self.optims['T'] = dict()
        for convert in self.test_converts:
            if self.args.task == 'clf':
                self.optims['T'][convert] = optim.SGD(self.nets['T'][convert].parameters(), lr=self.args.lr_clf, momentum=0.9,
                                                 weight_decay=self.args.weight_decay_task)
            elif self.args.task == 'seg':
                self.optims['T'][convert] = optim.SGD(self.nets['T'][convert].parameters(), lr=self.args.lr_seg, momentum=0.9,
                                         weight_decay=self.args.weight_decay_task)

    def set_zero_grad(self):
        for net in self.nets.keys():
            if net == 'D':
                for dset in self.args.datasets:
                    self.nets[net][dset].zero_grad()
            elif net == 'T':
                for convert in self.test_converts:
                    self.nets[net][convert].zero_grad()
            else:
                self.nets[net].zero_grad()

    def set_train(self):
        for net in self.nets.keys():
            if net == 'D':
                for dset in self.args.datasets:
                    self.nets[net][dset].train()
            elif net == 'T':
                for convert in self.test_converts:
                    self.nets[net][convert].train()
            else:
                self.nets[net].train()

    def set_eval(self):
        for convert in self.test_converts:
            self.nets['T'][convert].eval()

    def get_batch(self, batch_data_iter):
        batch_data = dict()
        for dset in self.args.datasets:
            try:
                batch_data[dset] = next(batch_data_iter[dset])
            except StopIteration:
                batch_data_iter[dset] = iter(self.train_loader[dset])
                batch_data[dset] = next(batch_data_iter[dset])
        return batch_data

    def get_grl_lambda(current_iter, max_iter, max_lambda=1.0):
        p = float(current_iter) / max_iter
        return max_lambda * (2. / (1. + np.exp(-10 * p)) - 1.)
        

    def train_dis(self, imgs):  # Train Discriminators (D)
        self.set_zero_grad()
        features, converted_imgs, D_outputs_fake, D_outputs_real = dict(), dict(), dict(), dict()
        classifier_outputs, combined = dict(), dict()

        # Real
        for dset in self.args.datasets:
            features[dset] = self.nets['E'](imgs[dset])
            if self.args.task == 'clf':
                ##################################################################
                if self.arg_use_D_optimization:
                    # 1. 获取分类器输出
                    if dset == 'M':
                        classifier_outputs[dset] = torch.softmax(self.nets['T']['M2MM'](imgs[dset]), dim=1)
                    elif dset == 'MM':
                        classifier_outputs[dset] = torch.softmax(self.nets['T']['MM2M'](imgs[dset]), dim=1)

                    # # 2. 计算联合表示 f(x) ⊗ p(y|x)
                    if self.arg_D_try1:
                        f = features[dset].to(self.device)                        # [B, C, H, W]
                        p = classifier_outputs[dset].to(self.device)              # [B, K]

                        combined = torch.einsum('bchw,bk->bkchw', f, p)           # [B, K, C, H, W]
                        combined = combined.view(combined.size(0), -1, *f.shape[2:])  # [B, K*C, H, W]
                        # 3. 用 Adapter 调整通道数 -> [B, 3, H, W]
                        combined = self.adapter[dset](combined)

                        # 4. 输入判别器
                        D_outputs_real[dset] = self.nets['D'][dset](combined)
                        # print('D_combined[dset].shape:', combined.shape)
                        # print('D_imgs[dset].shape:', imgs[dset].shape)
                    elif self.arg_D_try2:
                        f = features[dset].to(self.device)                        # [B, C, H, W]
                        p = classifier_outputs[dset].to(self.device)              # [B, K]

                        # print('f.shape:', f.shape)
                        # print('p.shape:', p.shape)

                        p_proj = self.p_projector[dset](p)                        # [B, C]
                        p_proj = p_proj.unsqueeze(-1).unsqueeze(-1)               # [B, C, 1, 1]
                        # print('p_proj.shape:', p_proj.shape)

                        gate = torch.sigmoid(self.gate[dset](f))                  # [B, 1, H, W]

                        combined[dset] = (1 - gate) * f + gate * (f * p_proj)     # [B, C, H, W]
                        # print('D_combined2[dset].shape:', combined[dset].shape)
                        combined[dset] = self.adapter[dset](combined[dset])       # [B, 3, H, W]
                        # print('D_combined1[dset].shape:', combined[dset].shape)
                        combined[dset] = F.interpolate(combined[dset], size=(64, 64), mode='bilinear', align_corners=False)

                        if self.arg_D_try3:
                            self.grl_lambda = get_grl_lambda(self.current_iter, self.total_iters)
                            combined[dset] = grad_reverse(combined[dset], lambda_=self.grl_lambda)

                        # print('D_combined[dset].shape:', combined[dset].shape)
                        # print('D_imgs[dset].shape:', imgs[dset].shape)
                        D_outputs_real[dset] = self.nets['D'][dset](combined[dset])

                    elif self.arg_D_try4:
                        f = features[dset].to(self.device)                        # [B, C, H, W]
                        p = classifier_outputs[dset].to(self.device)              # [B, K]

                        B, C, H, W = f.shape
                        K = p.shape[1]

                        # 1. 构造联合表示
                        joint = torch.einsum('bchw,bk->bkchw', f, p)              # [B, K, C, H, W]
                        joint = joint.view(B, K * C, H, W)                        # [B, K*C, H, W]
                        # 2. 展平 + 转置 -> [B, H*W, K*C]
                        joint_flat = joint.view(B, K * C, -1).permute(0, 2, 1)    # [B, H*W, K*C]
                        # 3. 线性降维 -> [B, H*W, low_dim]
                        joint_proj = self.rand_proj[dset](joint_flat)
                        # 4. reshape 成 [B, low_dim, H, W]
                        joint_proj = joint_proj.permute(0, 2, 1).contiguous().view(B, self.rand_low_dim, H, W)
                        # 5. 映射通道 + 判别器
                        combined = self.adapter[dset](joint_proj)                 # [B, 3, H, W]
                        # print('D_combined[dset].shape:', combined.shape)
                        # print('D_imgs[dset].shape:', imgs[dset].shape)
                        combined = F.interpolate(combined, size=(64, 64), mode='bilinear', align_corners=False)
                        # print('D_combined[dset].shape:', combined.shape)
                        D_outputs_real[dset] = self.nets['D'][dset](combined)

                ##################################################################
                else:
                    D_outputs_real[dset] = self.nets['D'][dset](imgs[dset])
            else:
                D_outputs_real[dset] = self.nets['D'][dset](slice_patches(imgs[dset]))

        contents, styles = self.nets['S'](features, self.training_converts)

        # CADT
        if self.args.CADT:
            for convert in self.training_converts:
                source, target = convert.split('2')
                _, styles[target] = cadt(contents[source], contents[target], styles[target])

        # Fake
        for convert in self.training_converts:
            source, target = convert.split('2')
            converted_imgs[convert] = self.nets['G'](contents[convert], styles[target])
            if self.args.task == 'clf':
                ##################################################################
                if self.arg_use_D_optimization:
                    # 1. 获取fake特征和分类概率
                    features_fake = self.nets['E'](converted_imgs[convert])  # [B, 64, H, W]
                    classifier_outputs_fake = torch.softmax(self.nets['T'][convert](converted_imgs[convert]), dim=1)  # [B, num_classes]
                    
                    if self.arg_D_try1:
                        # 2. 计算联合表示 (保持4D结构)
                        # 方法1：使用einsum保持空间结构
                        combined_fake = torch.einsum('bchw,bk->bkchw', features_fake, classifier_outputs_fake)  # [B, K, C, H, W]
                        combined_fake = combined_fake.view(combined_fake.size(0), -1, *combined_fake.shape[3:])  # [B, K*C, H, W]
                        
                        # 3. 通过适配层调整通道数
                        combined_fake = self.adapter[target](combined_fake)  # [B, 3, H, W]
                        
                        # 4. 输入判别器
                        D_outputs_fake[convert] = self.nets['D'][target](combined_fake)
                    elif self.arg_D_try2:
                        p_proj_fake = self.p_projector[target](classifier_outputs_fake)          # [B, C]
                        p_proj_fake = p_proj_fake.unsqueeze(-1).unsqueeze(-1)                    # [B, C, 1, 1]

                        gate_fake = torch.sigmoid(self.gate[target](features_fake))              # [B, 1, H, W]

                        combined_fake = (1 - gate_fake) * features_fake + gate_fake * (features_fake * p_proj_fake)
                        combined_fake = self.adapter[target](combined_fake)                      # [B, 3, H, W]
                        # print('combined_fake.shape:', combined_fake.shape)
                        combined_fake = F.interpolate(combined_fake, size=(64, 64), mode='bilinear', align_corners=False)
                        
                        if self.arg_D_try3:
                            self.grl_lambda = get_grl_lambda(self.current_iter, self.total_iters)
                            self.current_iter += 1
                            combined_fake = grad_reverse(combined_fake, lambda_=self.grl_lambda)

                        D_outputs_fake[convert] = self.nets['D'][target](combined_fake)
                    
                    elif self.arg_D_try4:
                        f = features_fake
                        p = classifier_outputs_fake
                        B, C, H, W = f.shape
                        K = p.shape[1]

                        joint = torch.einsum('bchw,bk->bkchw', f, p)              # [B, K, C, H, W]
                        joint = joint.view(B, K * C, H, W)                        # [B, K*C, H, W]
                        joint_flat = joint.view(B, K * C, -1).permute(0, 2, 1)    # [B, H*W, K*C]
                        joint_proj = self.rand_proj[target](joint_flat)           # [B, H*W, low_dim]
                        joint_proj = joint_proj.permute(0, 2, 1).contiguous().view(B, self.rand_low_dim, H, W)

                        combined_fake = self.adapter[target](joint_proj)          # [B, 3, H, W]
                        combined_fake = F.interpolate(combined_fake, size=(64, 64), mode='bilinear', align_corners=False)
                        D_outputs_fake[convert] = self.nets['D'][target](combined_fake)

                ##################################################################
                else:
                    D_outputs_fake[convert] = self.nets['D'][target](converted_imgs[convert])
            else:
                D_outputs_fake[convert] = self.nets['D'][target](slice_patches(converted_imgs[convert]))
                
        errD = self.loss_fns.dis(D_outputs_real, D_outputs_fake)
        errD.backward()
        for optimizer in self.optims['D'].values():
            optimizer.step()
        self.losses['D'] = errD.data.item()

    def train_task(self, imgs, labels):  # Train Task Networks (T)
        self.set_zero_grad()
        features = dict()
        converted_imgs = dict()
        pred = dict()
        converts = self.training_converts if self.args.task == 'clf' else self.test_converts
        with torch.no_grad():
            for dset in self.args.datasets:
                features[dset] = self.nets['E'](imgs[dset])
            contents, styles = self.nets['S'](features, converts)
            for convert in converts:
                source, target = convert.split('2')
                if self.args.CADT:
                    _, styles[target] = cadt(contents[source], contents[target], styles[target])
                converted_imgs[convert] = self.nets['G'](contents[convert], styles[target])

            # 3 datasets (MNIST, MNIST-M, USPS)
            # DRANet can convert USPS <-> MNIST-M without training the conversion directly.
            for convert in list(set(self.test_converts) - set(self.training_converts)):
                features_mid = dict()
                source, target = convert.split('2')
                mid = list(set(self.args.datasets) - {source, target})[0]
                convert1 = source + '2' + mid
                convert2 = mid + '2' + target
                features_mid[convert1] = self.nets['E'](converted_imgs[convert1])
                contents_mid, _ = self.nets['S'](features_mid, [convert2])
                converted_imgs[convert] = self.nets['G'](contents_mid[convert2], styles[target])

        for convert in self.test_converts:
            pred[convert] = self.nets['T'][convert](converted_imgs[convert])
            source, target = convert.split('2')
            pred[source] = self.nets['T'][convert](imgs[source])

        errT = self.loss_fns.task(pred, labels)
        errT.backward()
        for optimizer in self.optims['T'].values():
            optimizer.step()
        self.losses['T'] = errT.data.item()

    def train_esg(self, imgs):  # Train Encoder(E), Separator(S), Generator(G)
        self.set_zero_grad()
        features, converted_imgs, recon_imgs, D_outputs_fake = dict(), dict(), dict(), dict()
        features_converted = dict()
        perceptual, style_gram = dict(), dict()
        perceptual_converted, style_gram_converted = dict(), dict()
        con_sim = dict()
        preds = dict()  # 新增: 存储预测结果
        for dset in self.args.datasets:
            features[dset] = self.nets['E'](imgs[dset])
            recon_imgs[dset] = self.nets['G'](features[dset], 0)
            perceptual[dset] = self.nets['P'](imgs[dset])
            style_gram[dset] = [gram(fmap) for fmap in perceptual[dset][:-1]]
        contents, styles = self.nets['S'](features, self.training_converts)

        for convert in self.training_converts:
            source, target = convert.split('2')
            if self.args.CADT:
                con_sim[convert], styles[target] = cadt(contents[source], contents[target], styles[target])
                style_gram[target] = cadt_gram(style_gram[target], con_sim[convert])
            converted_imgs[convert] = self.nets['G'](contents[convert], styles[target])
            preds[convert] = self.nets['G'](contents[convert], styles[target])  # 新增：获取预测结果
            if self.args.task == 'clf':
                D_outputs_fake[convert] = self.nets['D'][target](converted_imgs[convert])
            else:
                D_outputs_fake[convert] = self.nets['D'][target](slice_patches(converted_imgs[convert]))
            features_converted[convert] = self.nets['E'](converted_imgs[convert])
            perceptual_converted[convert] = self.nets['P'](converted_imgs[convert])
            style_gram_converted[convert] = [gram(fmap) for fmap in perceptual_converted[convert][:-1]]
        contents_converted, styles_converted = self.nets['S'](features_converted)

        Content_loss = self.loss_fns.content_perceptual(perceptual, perceptual_converted)
        Style_loss = self.loss_fns.style_perceptual(style_gram, style_gram_converted)
        Consistency_loss = self.loss_fns.consistency(contents, styles, contents_converted, styles_converted, self.training_converts)
        G_loss = self.loss_fns.gen(D_outputs_fake)
        Recon_loss = self.loss_fns.recon(imgs, recon_imgs)

        # errESG = G_loss + Content_loss + Style_loss + Consistency_loss + Recon_loss
        ########################################################################################
        Entropy_loss = self.loss_fns.entropy(preds)  # 熵最小化损失
        VAT_loss = self.loss_fns.vat_loss(self.nets, converted_imgs)  # VAT损失
        alpha = 0
        beta = 0.01

        if self.arg_use_entropy == False:
            alpha = 0
        if self.arg_use_vat == False:
            beta = 0
            
        errESG = G_loss + Content_loss + Style_loss + Consistency_loss + Recon_loss \
                + alpha * Entropy_loss + beta * VAT_loss
        ########################################################################################

        errESG.backward()
        for net in ['E', 'S', 'G']:
            self.optims[net].step()

        self.losses['G'] = G_loss.data.item()
        self.losses['Recon'] = Recon_loss.data.item()
        self.losses['Consis'] = Consistency_loss.data.item()
        self.losses['Content'] = Content_loss.data.item()
        self.losses['Style'] = Style_loss.data.item()
        ########################################################################################
        self.losses['Entropy'] = Entropy_loss.data.item()
        self.losses['VAT'] = VAT_loss.data.item()
        ########################################################################################


    def tensor_board_log(self, imgs, labels):
        nrow = 8 if self.args.task == 'clf' else 2
        features, converted_imgs, recon_imgs = dict(), dict(), dict()
        converts = self.tensorboard_converts
        with torch.no_grad():
            for dset in self.args.datasets:
                features[dset] = self.nets['E'](imgs[dset])
                recon_imgs[dset] = self.nets['G'](features[dset], 0)
            contents, styles = self.nets['S'](features, self.training_converts)
            for convert in self.training_converts:
                source, target = convert.split('2')
                if self.args.CADT:
                    _, styles[target] = cadt(contents[source], contents[target], styles[target])
                converted_imgs[convert] = self.nets['G'](contents[convert], styles[target])
            # 3 datasets
            for convert in list(set(self.test_converts) - set(self.training_converts)):
                features_mid = dict()
                source, target = convert.split('2')
                mid = list(set(self.args.datasets) - {source, target})[0]
                convert1 = source + '2' + mid
                convert2 = mid + '2' + target
                features_mid[convert1] = self.nets['E'](converted_imgs[convert1])
                contents_mid, _ = self.nets['S'](features_mid, [convert2])
                converted_imgs[convert] = self.nets['G'](contents_mid[convert2], styles[target])

        # Input Images & Reconstructed Images
        for dset in self.args.datasets:
            x = vutils.make_grid(imgs[dset].detach(), normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('1_Input_Images/%s' % dset, x, self.step)
            x = vutils.make_grid(recon_imgs[dset].detach(), normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('2_Recon_Images/%s' % dset, x, self.step)

        # Converted Images
        for convert in converts:
            x = vutils.make_grid(converted_imgs[convert].detach(), normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('3_Converted_Images/%s' % convert, x, self.step)

        # Losses
        for loss in self.losses.keys():
            self.writer.add_scalar('Losses/%s' % loss, self.losses[loss], self.step)

        # Segmentation GT, Prediction
        if self.args.task == 'seg':
            vn = 2
            self.set_eval()
            preds = dict()
            for dset in self.args.datasets:
                x = decode_labels(labels[dset].detach(), num_images=vn)
                x = vutils.make_grid(x, normalize=True, scale_each=True, nrow=nrow)
                self.writer.add_image('4_GT/%s' % dset, x, self.step)
                preds[dset] = self.nets['T']['G2C'](imgs[dset])
            preds['G2C'] = self.nets['T']['G2C'](converted_imgs['G2C'])

            for key in preds.keys():
                pred = preds[key].data.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                x = decode_labels(pred, num_images=vn)
                x = vutils.make_grid(x, normalize=True, scale_each=True, nrow=nrow)
                self.writer.add_image('5_Prediction/%s' % key, x, self.step)
            self.set_train()

    def eval(self, cv):
        source, target = cv.split('2')
        self.set_eval()

        if self.args.task == 'clf':
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (imgs, labels) in enumerate(self.test_loader[target]):
                    imgs, labels = imgs.cuda(), labels.cuda()
                    pred = self.nets['T'][cv](imgs)
                    _, predicted = torch.max(pred.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    progress_bar(batch_idx, len(self.test_loader[target]), 'Acc: %.3f%% (%d/%d)'
                                 % (100. * correct / total, correct, total))
                # Save checkpoint.
                acc = 100. * correct / total
                self.logger.info('======================================================')
                self.logger.info('Step: %d | Acc: %.3f%% (%d/%d)'
                            % (self.step / len(self.test_loader[target]), acc, correct, total))
                self.logger.info('======================================================')
                self.writer.add_scalar('Accuracy/%s' % cv, acc, self.step)
                if acc > self.best_acc[cv]:
                    self.best_acc[cv] = acc
                    self.writer.add_scalar('Best_Accuracy/%s' % cv, acc, self.step)
                    self.save_networks()

        elif self.args.task == 'seg':
            miou = 0
            confusion_matrix = np.zeros((19,) * 2)
            with torch.no_grad():
                for batch_idx, (imgs, labels) in enumerate(self.test_loader[target]):
                    imgs, labels = imgs.cuda(), labels.cuda()
                    labels = labels.long()
                    pred = self.nets['T'][cv](imgs)
                    
                    pred = pred.data.cpu().numpy()
                    pred = np.argmax(pred, axis=1)
                    gt = labels.data.cpu().numpy()
                    confusion_matrix += MIOU(gt, pred)

                    score = np.diag(confusion_matrix) / (
                            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(
                        confusion_matrix))
                    miou = np.nanmean(score) * 100.

                    progress_bar(batch_idx, len(self.test_loader[target]), 'Acc: %.3f%%'
                                 % miou)
                score = 100 * np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix))
                score = np.round(score, 1)
                table = PrettyTable()
                table.field_names = self.name_classes_19
                table.add_row(score)
                # Save checkpoint.
                # miou = 100. * miou
                self.logger.info('======================================================')
                self.logger.info('Step: %d | mIoU: %.3f%%'
                            % (self.step, miou))
                self.logger.info(table)
                self.logger.info('======================================================')
                self.writer.add_scalar('MIoU/G2C', miou, self.step)
                if miou > self.best_miou:
                    self.best_miou = miou
                    self.writer.add_scalar('Best_MIoU/G2C', self.best_miou, self.step)
                    self.save_networks()
        self.set_train()

    def print_loss(self):
        best = ''
        if self.args.task == 'clf':
            for cv in self.test_converts:
                best = best + cv + ': %.2f' % self.best_acc[cv] + '|'
        elif self.args.task == 'seg':
            best += '%.2f' % self.best_miou

        losses = ''
        for key in self.losses:
            losses += ('%s: %.2f|'% (key, self.losses[key])) 
        self.logger.info(
            '[%d/%d] %s| %s %s'
            % (self.step, self.args.iter, losses, best, self.args.ex))

        # self.logger.info(
        #     '[%d/%d] D: %.2f | G: %.2f| R: %.2f| C: %.2f| S: %.2f| Cs: %.2f| T: %.2f| %s %s'
        #     % (self.step, self.args.iter,
        #        self.losses['D'], self.losses['G'], self.losses['Recon'], self.losses['Content'],
        #        self.losses['Style'], self.losses['Consis'], self.losses['T'], best, self.args.ex))

    def train(self):
        self.set_default()
        self.set_networks()
        self.set_optimizers()
        self.set_train()
        self.logger.info(self.loss_fns.alpha)
        batch_data_iter = dict()
        for dset in self.args.datasets:
            batch_data_iter[dset] = iter(self.train_loader[dset])

        for i in range(self.args.iter):
            self.step += 1
            # get batch data
            batch_data = self.get_batch(batch_data_iter)
            imgs, labels = dict(), dict()
            min_batch = self.args.batch
            for dset in self.args.datasets:
                imgs[dset], labels[dset] = batch_data[dset]
                imgs[dset], labels[dset] = imgs[dset].cuda(), labels[dset].cuda()
                if self.args.task == 'seg':
                    labels[dset] = labels[dset].long()
                if imgs[dset].size(0) < min_batch:
                    min_batch = imgs[dset].size(0)
            if min_batch < self.args.batch:
                for dset in self.args.datasets:
                    imgs[dset], labels[dset] = imgs[dset][:min_batch], labels[dset][:min_batch]
            # training
            self.train_dis(imgs)
            for t in range(2):
                self.train_esg(imgs)
            self.train_task(imgs, labels)
            # tensorboard
            if self.step % self.args.tensor_freq == 0:
                self.tensor_board_log(imgs, labels)
            # evaluation
            if self.step % self.args.eval_freq == 0:
                for cv in self.test_converts:
                    self.eval(cv)
            self.print_loss()

    def test(self):
        self.set_default()
        self.set_networks()
        self.load_networks(self.args.load_step)
        for cv in self.test_converts:
            self.eval(cv)

