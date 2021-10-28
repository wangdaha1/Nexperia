import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from misc import remix_lam

# 定义一些基本的模块
class Transpose(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        """
        (B, S, C) or (B, C, S)
        """
        return x.transpose(1, 2)

class MlpBlock(nn.Module):
    """
    linear works on the last dim
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(), # 错了 应该是nn.GELU()
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.mlp(x)

class MixerBlock(nn.Module):
    def __init__(self, token_dim, channel_dim):
        super().__init__()
        self.token_mixing = nn.Sequential(
            nn.LayerNorm([token_dim, channel_dim]),
            Transpose(),
            MlpBlock(token_dim),
            Transpose(),
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm([token_dim, channel_dim]),
            MlpBlock(channel_dim),
        )

    def forward(self, x):
        x = self.token_mixing(x) + x
        return x + self.channel_mixing(x)

class StratifiedPatch(nn.Module):
    def __init__(self, channel_dim, fig_size):
        super().__init__()
        self.channel_dim = channel_dim
        self.width, self.height = fig_size
        # slice the image three times, these kernel sizes are fixed here. You can also adapt them
        self.p_32 = nn.Conv2d(3, channel_dim, kernel_size=(32, 32), stride=(32, 32))
        self.p_vertical = nn.Conv2d(3, channel_dim, kernel_size=(16, 128), stride=(16, 128))
        self.p_horizontal = nn.Conv2d(3, channel_dim, kernel_size=(128, 16), stride=(128, 16))

    def forward(self, x):
        # x1 = self.p_16(x).view(x.size(0), self.channel_dim, -1)
        x1 = self.p_32(x).view(x.size(0), self.channel_dim, -1)
        # x3 = self.p_64(x).view(x.size(0), self.channel_dim, -1)
        x2 = self.p_vertical(x).view(x.size(0), self.channel_dim, -1)
        x3 = self.p_horizontal(x).view(x.size(0), self.channel_dim, -1)
        return torch.cat([x1, x2, x3], dim=2)

    def __get_token_dim(self):
        t_dim = sum([(self.width // p_s) * (self.height // p_s) for p_s in [32]])
        t_dim += sum([(self.width // p_x) * (self.height // p_y) for (p_x, p_y) in [(16, 128), (128, 16)]])
        return t_dim

    token_dim = property(__get_token_dim)

# MLP-Mixer model
class MLPMixer(nn.Module):
    def __init__(self, patch_size: int, channel_dim: int, num_blocks: int, fig_size):
        super().__init__()
        self.patch_size = patch_size
        # self.token_dim = sum([(width // p_s) * (height // p_s) for p_s in [16, 32, 64]])
        self.channel_dim = channel_dim
        self.num_blocks = num_blocks
        if self.patch_size < 0:
            self.patch_proj = StratifiedPatch(self.channel_dim, fig_size)
            self.token_dim = self.patch_proj.token_dim
        else:
            self.patch_proj = nn.Conv2d(3, channel_dim, kernel_size=(patch_size, patch_size),
                                        stride=(patch_size, patch_size))
            width, height = fig_size
            self.token_dim = (width // patch_size) * (height // patch_size)

        layers = [MixerBlock(self.token_dim, self.channel_dim) for _ in range(num_blocks)]
        self.mixer_mlp_blocks = nn.Sequential(*layers)
        self.out_LayerNorm = nn.LayerNorm([self.token_dim, self.channel_dim])
        self.out_fc = nn.Linear(self.channel_dim, 2)  # 可以直接在这里把2改成num_classes作为参数输入

    def forward(self, x):
        x = self.patch_proj(x).view(x.size(0), self.channel_dim, -1).transpose(1, 2)
        x = self.mixer_mlp_blocks(x)
        x = self.out_LayerNorm(x)
        x = self.out_fc(x.mean(axis=1))  # global avg. pooling

        return x

# mainfold mixup
def to_one_hot(inp, num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes).to(inp.device)
    y_onehot.zero_()
    y_onehot.scatter_(1, inp.unsqueeze(1).data, 1)
    return y_onehot

class ManifoldMixupModel(nn.Module):
    '''
    这里是直接用的https://blog.csdn.net/Brikie/article/details/114222605的代码
    里面默认的是用了resnet网络  CE损失函数
    如果要改的话需要在这里面改动的
    '''
    def __init__(self, model, num_classes, alpha):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.lam = None
        self.num_classes = num_classes
        # 选择需要操作的层，在ResNet中各block的层名为layer1,layer2...所以可以写成如下。其他网络请自行修改
        # 也就是说 现在这个函数还只支持用resnet网络
        self.module_list = []
        for n, m in self.model.named_modules():
            # if 'conv' in n:
            if n[:-1] == 'layer':
                self.module_list.append(m)

    def forward(self, x, target=None):
        if target == None: # 这里就是用于模型的validation的 在训练模型的时候都是model(inputs, labels) valid的时候是model(inputs)
            out = self.model(x)  # 所以ModelTrainer.valid是不需要做任何变化的
            return out
        else:
            if self.alpha <= 0:
                self.lam = 1
            else:
                self.lam = np.random.beta(self.alpha, self.alpha)
            k = np.random.randint(-1, len(self.module_list))
            # k = len(self.module_list)-3
            self.indices = torch.randperm(target.size(0)).cuda()
            target_onehot = to_one_hot(target, self.num_classes)
            target_shuffled_onehot = target_onehot[self.indices]
            if k == -1:
                x = x * self.lam + x[self.indices] * (1 - self.lam)
                out = self.model(x)
            else:
                modifier_hook = self.module_list[k].register_forward_hook(self.hook_modify)
                out = self.model(x)
                modifier_hook.remove()
            target_reweighted = target_onehot * self.lam + target_shuffled_onehot * (1 - self.lam)

            # loss = nn.CrossEntropyLoss(out, target_reweighted)
            # loss = bce_loss(softmax(out), target_reweighted)
            target_a = torch.topk(target_onehot, 1)[1].squeeze(1)
            target_b = torch.topk(target_shuffled_onehot, 1)[1].squeeze(1)
            # 我晕。。。这也太坑了吧  从没注意过
            loss = self.lam*nn.CrossEntropyLoss()(out, target_a) + (1-self.lam)*nn.CrossEntropyLoss()(out, target_b)
            return out, loss

    def hook_modify(self, module, input, output):
        output = self.lam * output + (1 - self.lam) * output[self.indices]
        return output

# patchup
class PatchUpModel(nn.Module):
    '''
    这里是直接用的https://blog.csdn.net/Brikie/article/details/114222605的代码
    用了resnet网络  CE损失函数
    '''
    def __init__(self, model, num_classes=9, block_size=7, gamma=.9, patchup_type='hard'):
        super().__init__()
        self.patchup_type = patchup_type
        self.block_size = block_size
        self.gamma = gamma
        print("block size:", self.block_size, "gamma:", gamma)
        self.gamma_adj = None
        self.kernel_size = (block_size, block_size)
        self.stride = (1, 1)
        self.padding = (block_size // 2, block_size // 2)
        self.computed_lam = None

        self.model = model
        self.num_classes = num_classes
        self.module_list = []
        for n, m in self.model.named_modules():
            if n[:-1] == 'layer':
                # if 'conv' in n:
                self.module_list.append(m)

    def adjust_gamma(self, x):
        return self.gamma * x.shape[-1] ** 2 / \
               (self.block_size ** 2 * (x.shape[-1] - self.block_size + 1) ** 2)

    def forward(self, x, target=None):
        if target == None:
            out = self.model(x)
            return out
        else:

            self.lam = np.random.beta(2.0, 2.0)
            k = np.random.randint(-1, len(self.module_list))
            self.indices = torch.randperm(target.size(0)).cuda()
            self.target_onehot = to_one_hot(target, self.num_classes)
            self.target_shuffled_onehot = self.target_onehot[self.indices]
            self.target_onehot_ce = torch.topk(self.target_onehot, 1)[1].squeeze(1)
            self.target_shuffled_onehot_ce = torch.topk(self.target_shuffled_onehot, 1)[1].squeeze(1)

            if k == -1:  # CutMix
                W, H = x.size(2), x.size(3)
                cut_rat = np.sqrt(1. - self.lam)
                cut_w = np.int(W * cut_rat)
                cut_h = np.int(H * cut_rat)
                cx = np.random.randint(W)
                cy = np.random.randint(H)

                bbx1 = np.clip(cx - cut_w // 2, 0, W)
                bby1 = np.clip(cy - cut_h // 2, 0, H)
                bbx2 = np.clip(cx + cut_w // 2, 0, W)
                bby2 = np.clip(cy + cut_h // 2, 0, H)

                x[:, :, bbx1:bbx2, bby1:bby2] = x[self.indices, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
                out = self.model(x)
                loss  = lam*nn.CrossEntropyLoss()(out, self.target_onehot_ce) + \
                        (1-lam)*nn.CrossEntropyLoss()(out, self.target_shuffled_onehot_ce)
                # loss = bce_loss(softmax(out), self.target_onehot) * lam + \
                #        bce_loss(softmax(out), self.target_shuffled_onehot) * (1. - lam)

            else:
                modifier_hook = self.module_list[k].register_forward_hook(self.hook_modify)
                out = self.model(x)
                modifier_hook.remove()

                if self.patchup_type == 'hard':
                    loss = nn.CrossEntropyLoss()(out, self.target_onehot_ce) * self.total_unchanged_portion +\
                           (nn.CrossEntropyLoss()(out, self.target_shuffled_onehot_ce))*(1. - self.total_unchanged_portion)+\
                           (nn.CrossEntropyLoss()(out, self.target_onehot_ce)*self.total_unchanged_portion+\
                            nn.CrossEntropyLoss()(out, self.target_shuffled_onehot_ce)*self.total_changed_portion)
                elif self.patchup_type == 'soft':
                    loss = nn.CrossEntropyLoss()(out, self.target_onehot_ce) * self.total_unchanged_portion +\
                           (self.lam*nn.CrossEntropyLoss()(out, self.target_onehot_ce)+\
                            (1-self.lam)*nn.CrossEntropyLoss()(out, self.target_shuffled_onehot_ce))*(1. - self.total_unchanged_portion)+\
                           (self.total_unchanged_portion*nn.CrossEntropyLoss()(out, self.target_onehot_ce)+\
                            self.lam*self.total_changed_portion*nn.CrossEntropyLoss()(out, self.total_changed_portion)+
                            (1-self.lam)*self.total_changed_portion*nn.CrossEntropyLoss()(out,self.target_shuffled_onehot_ce))
                # loss = 1.0 * nn.CrossEntropyLoss()(out, self.target_onehot) * self.total_unchanged_portion + \
                #        nn.CrossEntropyLoss()(out, self.target_b)* (1. - self.total_unchanged_portion) + \
                #        1.0 * nn.CrossEntropyLoss()(out, self.target_reweighted)
            return out, loss

    def hook_modify(self, module, input, output):
        self.gamma_adj = self.adjust_gamma(output)
        p = torch.ones_like(output[0]) * self.gamma_adj
        m_i_j = torch.bernoulli(p)
        mask_shape = len(m_i_j.shape)
        m_i_j = m_i_j.expand(output.size(0), m_i_j.size(0), m_i_j.size(1), m_i_j.size(2))
        holes = F.max_pool2d(m_i_j, self.kernel_size, self.stride, self.padding)
        mask = 1 - holes
        unchanged = mask * output
        if mask_shape == 1:
            total_feats = output.size(1)
        else:
            total_feats = output.size(1) * (output.size(2) ** 2)
        total_changed_pixels = holes[0].sum()
        self.total_changed_portion = total_changed_pixels / total_feats
        self.total_unchanged_portion = (total_feats - total_changed_pixels) / total_feats
        if self.patchup_type == 'hard':
            self.target_reweighted = self.total_unchanged_portion * self.target_onehot + \
                                     self.total_changed_portion * self.target_shuffled_onehot
            patches = holes * output[self.indices]
            self.target_b = self.target_onehot[self.indices]
            self.target_b_ce = torch.topk(self.target_b, 1)[1].squeeze(1)
        elif self.patchup_type == 'soft':
            self.target_reweighted = self.total_unchanged_portion * self.target_onehot + \
                                     self.lam * self.total_changed_portion * self.target_onehot + \
                                     (1 - self.lam) * self.total_changed_portion * self.target_shuffled_onehot
            patches = holes * output
            patches = patches * self.lam + patches[self.indices] * (1 - self.lam)
            self.target_b = self.lam * self.target_onehot + (1 - self.lam) * self.target_shuffled_onehot
        else:
            raise ValueError("patchup_type must be \'hard\' or \'soft\'.")

        output = unchanged + patches
        self.target_a = self.target_onehot
        return output

class Remix_ManifoldMixupModel(nn.Module):
    def __init__(self, model, num_classes, alpha, kappa, tau):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.kappa = kappa
        self.tau = tau
        self.num_classes = num_classes
        self.module_list = []
        for n, m in self.model.named_modules():
            # if 'conv' in n:
            if n[:-1] == 'layer':
                self.module_list.append(m)

    def forward(self, x, target=None):
        if target == None:
            out = self.model(x)
            return out
        else:
            if self.alpha <= 0:
                self.lam_original = 1
            else:
                self.lam_original = np.random.beta(self.alpha, self.alpha)
            k = np.random.randint(-1, len(self.module_list))
            # k = len(self.module_list)-3
            self.indices = torch.randperm(target.size(0)).cuda()
            target_onehot = to_one_hot(target, self.num_classes)
            target_shuffled_onehot = target_onehot[self.indices]
            label_i=target.cpu().numpy().tolist()
            label_j = torch.topk(target_shuffled_onehot, 1)[1].squeeze(1).cpu().numpy().tolist()
            lam_list = remix_lam(label_i, label_j, self.kappa, self.lam_original, self.tau)
            self.lam_list = torch.Tensor(lam_list).to('cuda')
            if k == -1:
                x = x * self.lam_original + x[self.indices] * (1 - self.lam_original)
                out = self.model(x)
            else:
                modifier_hook = self.module_list[k].register_forward_hook(self.hook_modify)
                out = self.model(x)
                modifier_hook.remove()

            # target_reweighted = target_onehot * self.lam_list + target_shuffled_onehot * (1 - self.lam_list)
            # loss = bce_loss(softmax(out), target_reweighted)
            target_a = torch.topk(target_onehot, 1)[1].squeeze(1)
            target_b = torch.topk(target_shuffled_onehot, 1)[1].squeeze(1)
            loss = 0
            for i in range(0, len(target_a)):
                loss += self.lam_list[i]*nn.CrossEntropyLoss()(out[i].unsqueeze(0), target_a[i].unsqueeze(0))+(1-self.lam_list[i])*nn.CrossEntropyLoss()(out[i].unsqueeze(0), target_b[i].unsqueeze(0))
            return out, loss

    def hook_modify(self, module, input, output):
        # output = self.lam_list*output+(1-self.lam_list)*output[self.indices]
        # output = self.lam_original * output + (1 - self.lam_original) * output[self.indices]
        output = self.lam_list.reshape((len(self.lam_list),1,1,1))*output+(1-self.lam_list.reshape((len(self.lam_list),1,1,1)))*output[self.indices]
        return output

