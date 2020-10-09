import torch.nn as nn
import torch
import numpy as np
import models
import torch.nn.functional as F
import torchvision
import os
import torchvision.transforms as transforms
from torch.autograd import Variable


def flat_trans(x):
    x.resize_(28*28)
    return x

# 概率向量重排序函数
def probability_vector_rerank(orig_vector, target):
    prob_vector = orig_vector.clone()
    prob_vector[ : , target] = 3
    return F.softmax(prob_vector, dim = 1)


# def cross_entro(prob_model, target_rank):

models_path = './models5/'


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class AdvGAN_Attack:
    def __init__(self,
                 device,
                 model,
                 model_num_labels,
                 image_nc,
                 box_min,
                 box_max):
        output_nc = image_nc
        self.device = device
        self.model_num_labels = model_num_labels
        self.model = model
        self.input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max

        self.gen_input_nc = image_nc
        self.netG = models.Generator(self.gen_input_nc, image_nc).to(device)
        self.netDisc = models.Discriminator(image_nc).to(device)

        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=0.5)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=0.5)
        # self.optimizer_G = torch.optim.SGD(self.netG.parameters(),
        #                                     lr=0.1, momentum=0.9)
        # self.optimizer_D = torch.optim.SGD(self.netDisc.parameters(),
        #                                     lr=0.1,momentum=0.9)

        if not os.path.exists(models_path):
            os.makedirs(models_path)

    def train_batch(self, x, labels):
        # optimize D
        for i in range(1):
            perturbation = self.netG(x)
            # modify step3
            orig_model = self.model(x)
            # print(F.softmax(orig_model, 1))
            # add a clipping trick
            adv_images = torch.clamp(perturbation, -0.3, 0.3) + x
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

            self.optimizer_D.zero_grad()
            pred_real = self.netDisc(x)
            loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
            loss_D_real.backward()

            pred_fake = self.netDisc(adv_images.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            loss_D_fake.backward()
            loss_D_GAN = loss_D_fake + loss_D_real
            self.optimizer_D.step()

        # optimize G
        for i in range(1):
            self.optimizer_G.zero_grad()

            # cal G's loss in GAN
            pred_fake = self.netDisc(adv_images)
            loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            loss_G_fake.backward(retain_graph=True)

            # calculate perturbation norm
            C = 0.1
            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
            # modify step1
            loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))

            # print(adv_images.view(2,784).shape)
            # print(mnist_transform(adv_images.data).shape)
            # cal adv loss
            logits_model = self.model(adv_images)
            # print('logits_model:',logits_model)
            probs_model = F.softmax(logits_model, dim=-1)
            print(probs_model[0].sum(),probs_model[1])
            # print('probs_model:', probs_model.shape)
            # modify step2
            # onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]
            # # print('labels:',labels)
            # # print('onehot_labels',onehot_labels)
            # # C&W loss function
            # real = torch.sum(onehot_labels * probs_model, dim=1)
            # # print('real', real)
            # other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            # zeros = torch.zeros_like(other)
            # loss_adv = torch.max(real - other, zeros)
            # # print('Before',loss_adv.shape)
            # loss_adv = torch.sum(loss_adv)
            # print('After', loss_adv.shape)
            # print(orig_model[0])
            # print(F.softmax(orig_model, 1)[0])
            target_rank = probability_vector_rerank(F.softmax(orig_model, 1), 4)
            # print(target_rank[0])
            # print(probs_model.shape, target_rank.shape)
            # print(labels)
            loss_adv = F.binary_cross_entropy(probs_model, target_rank.detach())

            # maximize cross_entropy loss
            # loss_adv = -F.mse_loss(logits_model, onehot_labels)
            # loss_adv = - F.cross_entropy(logits_model, labels)

            adv_lambda = 10
            pert_lambda = 1
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
            loss_G.backward()
            self.optimizer_G.step()

        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item()

    def train(self, train_dataloader, epochs):
        for epoch in range(1, epochs+1):

            if epoch == 40:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.1)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.1)
                # self.optimizer_G = torch.optim.SGD(self.netG.parameters(),
                #                                    lr=0.01, momentum=0.9)
                # self.optimizer_D = torch.optim.SGD(self.netDisc.parameters(),
                #                                    lr=0.01, momentum=0.9)
            if epoch == 800:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.01)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.01)
                # self.optimizer_G = torch.optim.SGD(self.netG.parameters(),
                #                                    lr=0.001, momentum=0.9)
                # self.optimizer_D = torch.optim.SGD(self.netDisc.parameters(),
                #                                    lr=0.001, momentum=0.9)
            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch = \
                    self.train_batch(images, labels)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch

            # print statistics
            num_batch = len(train_dataloader)
            print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
             \nloss_perturb: %.3f, loss_adv: %.3f, \n" %
                  (epoch, loss_D_sum/num_batch, loss_G_fake_sum/num_batch,
                   loss_perturb_sum/num_batch, loss_adv_sum/num_batch))

            # save generator
            if epoch%20==0:
                netG_file_name = models_path + 'netG_epoch_' + str(epoch) + '.pth'
                torch.save(self.netG.state_dict(), netG_file_name)

