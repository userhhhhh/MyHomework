import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_weights(task, dsets):
    '''
        You must set these hyperparameters to apply our method to other datasets.
        These hyperparameters may not be the optimal value for your machine.
    '''

    alpha = dict()
    alpha['style'], alpha['dis'], alpha['gen'] = dict(), dict(), dict()
    if task == 'clf':
        alpha['recon'], alpha['consis'], alpha['content'] = 5, 1, 1

        # MNIST <-> MNIST-M
        if 'M' in dsets and 'MM' in dsets and 'U' not in dsets:
            alpha['style']['M2MM'], alpha['style']['MM2M'] = 5e4, 1e4
            alpha['dis']['M'], alpha['dis']['MM'] = 0.5, 0.5
            alpha['gen']['M'], alpha['gen']['MM'] = 0.5, 1.0
            alpha['entropy'] = 0.01 # 新增
            alpha['vat'] = 0.01 # 新增

        # MNIST <-> USPS
        elif 'M' in dsets and 'U' in dsets and 'MM' not in dsets:
            alpha['style']['M2U'], alpha['style']['U2M'] = 5e3, 5e3
            alpha['dis']['M'], alpha['dis']['U'] = 0.5, 0.5
            alpha['gen']['M'], alpha['gen']['U'] = 0.5, 0.5

        # MNIST <-> MNIST-M <-> USPS
        elif 'M' in dsets and 'U' in dsets and 'MM' in dsets:
            alpha['style']['M2MM'], alpha['style']['MM2M'], alpha['style']['M2U'], alpha['style']['U2M'] = 5e4, 1e4, 1e4, 1e4
            alpha['dis']['M'], alpha['dis']['MM'], alpha['dis']['U'] = 0.5, 0.5, 0.5
            alpha['gen']['M'], alpha['gen']['MM'], alpha['gen']['U'] = 0.5, 1.0, 0.5

    elif task == 'seg':
        # GTA5 <-> Cityscapes
        alpha['recon'], alpha['consis'], alpha['content'] = 10, 1, 1
        alpha['style']['G2C'], alpha['style']['C2G'] = 5e3, 5e3
        alpha['dis']['G'], alpha['dis']['C'] = 0.5, 0.5
        alpha['gen']['G'], alpha['gen']['C'] = 0.5, 0.5

    return alpha


class Loss_Functions:
    def __init__(self, args):
        self.args = args
        self.alpha = loss_weights(args.task, args.datasets)
        ###################################################################

        # print("\nCurrent alpha weights:")  # 打印标题
        # for key, value in self.alpha.items():
        #     if isinstance(value, dict):  # 如果值是字典，进一步展开
        #         print(f"{key}:")
        #         for sub_key, sub_value in value.items():
        #             print(f"  {sub_key}: {sub_value}")
        #     else:
        #         print(f"{key}: {value}")

 ##########################################################################
    # def entropy(self, pred):
    #     """
    #     计算目标域（无标签）预测的 entropy loss。
    #     适用于 pred 是 softmax logits 字典，键是数据集名（例如 'MM'）。
    #     """
    #     ent_loss = 0
    #     for dset in pred.keys():
    #         if dset == 'MM':  # 只对 MNIST-M（目标域）使用 entropy loss
    #             probs = F.softmax(pred[dset], dim=1)
    #             ent = -torch.sum(probs * torch.log(probs + 1e-6), dim=1).mean()
    #             ent_loss += ent
    #     return 0.01 * ent_loss  # α 可调

    # def vat(self, model, x, xi=10.0, eps=1.0, ip=1):
    #     """
    #     计算 VAT loss（虚拟对抗训练）用于目标域样本。
    #     model: 分类器，输入 x 输出 logits
    #     x: 目标域输入图像（如 MNIST-M 图像）
    #     """
    #     with torch.no_grad():
    #         pred = F.softmax(model(x), dim=1)

    #     d = torch.randn_like(x)
    #     d = F.normalize(d, p=2, dim=(1, 2, 3))

    #     for _ in range(ip):
    #         d.requires_grad_()
    #         pred_hat = F.log_softmax(model(x + xi * d), dim=1)
    #         loss = F.kl_div(pred_hat, pred, reduction='batchmean')
    #         grad = torch.autograd.grad(loss, d, retain_graph=True)[0]
    #         d = F.normalize(grad, p=2, dim=(1, 2, 3))

    #     r_adv = eps * d
    #     pred_hat = F.log_softmax(model(x + r_adv), dim=1)
    #     vat_loss = F.kl_div(pred_hat, pred, reduction='batchmean')
    #     return 0.1 * vat_loss  # β 可调
    def entropy(self, pred):
        """
        计算目标域的熵最小化损失
        pred: 目标域的预测概率分布 (softmax输出)
        """
        entropy_loss = 0
        for key in pred.keys():
            if '2' in key:  # 只对转换后的目标域样本计算
                p = F.softmax(pred[key], dim=1)
                log_p = F.log_softmax(pred[key], dim=1)
                entropy_loss += (-p * log_p).sum(1).mean()  # 计算熵并取平均
        return self.alpha['entropy'] * entropy_loss
    
    def vat_loss(self, model, imgs, epsilon=0.3, xi=0.03, iterations=1):
        """
        计算虚拟对抗训练(VAT)损失
        model: 包含E,S,G的网络
        imgs: 输入图像
        epsilon: 扰动大小
        xi: 扰动计算步长
        iterations: 计算扰动的迭代次数
        """
        vat_loss = 0
        for dset in imgs.keys():
            if '2' in dset:  # 只对目标域样本计算
                x = imgs[dset]
                # x.requires_grad = True
                x_adv = x.clone().detach().requires_grad_(True)  # question:创建可求导的副本
                
                # 1. 计算原始预测
                with torch.no_grad():
                    features = model['E'](x)
                    contents, styles = model['S']({dset: features})
                    # print("Styles keys:", styles.keys())  # 检查是否包含 'MM'
                    pred = model['G'](contents[dset], styles[dset])
                    p = F.softmax(pred, dim=1)
                
                # 2. 计算初始随机扰动
                d = torch.randn_like(x_adv)
                d = d / (torch.norm(d, p=2) + 1e-8)
                
                # 3. 迭代计算对抗扰动
                for _ in range(iterations):
                    d.requires_grad = True
                    x_hat = x_adv + xi * d
                    features_hat = model['E'](x_hat)
                    contents_hat, styles_hat = model['S']({dset: features_hat})
                    pred_hat = model['G'](contents_hat[dset], styles_hat[dset]) # question:dset
                    p_hat = F.softmax(pred_hat, dim=1)
                    
                    kl_div = F.kl_div(F.log_softmax(p_hat, dim=1), p, reduction='batchmean')
                    kl_div.backward(retain_graph=True)  # 保留计算图
                    
                    d = xi * d.grad.data
                    d = d / (torch.norm(d, p=2) + 1e-8)
                    model['G'].zero_grad() # question:清除梯度                
                # 4. 计算最终VAT损失
                x_hat = x_adv + epsilon * d
                features_hat = model['E'](x_hat)
                contents_hat, styles_hat = model['S']({dset: features_hat})
                pred_hat = model['G'](contents_hat[dset], styles_hat[dset])
                p_hat = F.softmax(pred_hat, dim=1)
                
                vat_loss += F.kl_div(F.log_softmax(p_hat, dim=1), p, reduction='batchmean')
        
        return self.alpha['vat'] * vat_loss
 ##########################################################################

    def recon(self, imgs, recon_imgs):
        recon_loss = 0
        for dset in imgs.keys():
            recon_loss += F.l1_loss(imgs[dset], recon_imgs[dset])
        return self.alpha['recon'] * recon_loss
        
    def dis(self, real, fake):
        dis_loss = 0
        if self.args.task == 'clf':  # DCGAN loss
            for dset in real.keys():
                dis_loss += self.alpha['dis'][dset] * F.binary_cross_entropy(real[dset], torch.ones_like(real[dset]))
            for cv in fake.keys():
                source, target = cv.split('2')
                dis_loss += self.alpha['dis'][target] * F.binary_cross_entropy(fake[cv], torch.zeros_like(fake[cv]))
        elif self.args.task == 'seg':  # Hinge loss
            for dset in real.keys():
                dis_loss += self.alpha['dis'][dset] * F.relu(1. - real[dset]).mean()
            for cv in fake.keys():
                source, target = cv.split('2')
                dis_loss += self.alpha['dis'][target] * F.relu(1. + fake[cv]).mean()
        return dis_loss

    def gen(self, fake):
        gen_loss = 0
        for cv in fake.keys():
            source, target = cv.split('2')
            if self.args.task == 'clf':
                gen_loss += self.alpha['gen'][target] * F.binary_cross_entropy(fake[cv], torch.ones_like(fake[cv]))
            elif self.args.task == 'seg':
                gen_loss += -self.alpha['gen'][target] * fake[cv].mean()
        return gen_loss

    def content_perceptual(self, perceptual, perceptual_converted):
        content_perceptual_loss = 0
        for cv in perceptual_converted.keys():
            source, target = cv.split('2')
            content_perceptual_loss += F.mse_loss(perceptual[source][-1], perceptual_converted[cv][-1])
        return self.alpha['content'] * content_perceptual_loss

    def style_perceptual(self, style_gram, style_gram_converted):
        style_percptual_loss = 0
        for cv in style_gram_converted.keys():
            source, target = cv.split('2')
            for gr in range(len(style_gram[target])):
                style_percptual_loss += self.alpha['style'][cv] * F.mse_loss(style_gram[target][gr], style_gram_converted[cv][gr])
        return style_percptual_loss

    def consistency(self, contents, styles, contents_converted, styles_converted, converts):
        consistency_loss = 0
        for cv in converts:
            source, target = cv.split('2')
            consistency_loss += F.l1_loss(contents[cv], contents_converted[cv])
            consistency_loss += F.l1_loss(styles[target], styles_converted[cv])
        return self.alpha['consis'] * consistency_loss

    def task(self, pred, gt):
        task_loss = 0
        for key in pred.keys():
            if '2' in key:
                source, target = key.split('2')
            else:
                source = key
            task_loss += F.cross_entropy(pred[key], gt[source], ignore_index=-1)
        return task_loss

