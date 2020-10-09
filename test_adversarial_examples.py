import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import models
from models import MNIST_target_net
from train_mnist import Net
import os, sys, pickle
import matplotlib.pyplot as plt
import matplotlib.image as mping
from tqdm import *
import numpy as np
import time

def flat_trans(x):
    x.resize_(28*28)
    return x


def select_tensor(a, b):
    c = []
    num = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            a[i] = b[i]
        else:
            c.append([a[i].cpu().data,b[i].cpu().data])
            if a[i].cpu().data == 2 and b[i].cpu().data != 2:
                num += 1
    return num



mnist_transform = transforms.Lambda(flat_trans)


if __name__ == '__main__':
    use_cuda=True
    image_nc=1
    batch_size = 128
    totalMisclassfications = 0
    gen_input_nc = image_nc
    num_success = 0
    adv_examples = []
    # Define what device we are using
    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # # Load 5k samples
    # pkl_loc = './5k_samples.pkl'
    #     samples_5k = pickle.load(f)
    # xs = samples_5k["images"]
    # y_trues = samples_5k["labels"]
    #
    # # Aggregate
    # y_preds, y_preds_adversarial, noises = [], [], []
    # xs_clean = []
    # y_trues_clean = []

    # load the pretrained model
    pretrained_model = "./target_model_fcn.pth"
    target_model = Net().to(device)
    target_model.load_state_dict(torch.load(pretrained_model))
    target_model.eval()

    # load the CNN model
    # pretrained_model = "./MNIST_target_model.pth"
    # target_model = MNIST_target_net().to(device)
    # target_model.load_state_dict(torch.load(pretrained_model))
    # target_model.eval()

    # load the generator of adversarial examples
    pretrained_generator_path = './models3/netG_epoch_400.pth'
    pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
    pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
    pretrained_G.eval()
    # print(len(xs))
    # for x, y_true in tqdm(zip(xs, y_trues)):
    #
    #     x = x.unsqueeze(0)
    #     x, y_true = x.to(device), y_true.to(device)
    #     # Genrate Adeversarial Image
    #     perturbation = pretrained_G(x)
    #     perturbation = torch.clamp(perturbation, -0.3, 0.3)
    #     adv_img = perturbation + x
    #     y_pred = torch.argmax(target_model(x), -1).cpu().data.numpy()
    #     y_pred_adversarial = torch.argmax(target_model(adv_img), -1).cpu().data.numpy()
    #     if y_true.cpu().numpy() != y_pred:
    #         totalMisclassfications += 1
    #     else:
    #         y_preds.append(y_pred)
    #         y_preds_adversarial.append(y_pred_adversarial)
    #         noises.append(perturbation.cpu().detach().numpy())
    #         xs_clean.append(x.cpu().data.numpy())
    #         y_trues_clean.append(y_true.cpu().data.numpy())
    # print("Total totalMisclassifications : ", totalMisclassfications)
    # print("out of : ", len(xs))
    #
    # with open("bulk_advgan2.pkl", "wb") as f:
    #     adv_data_dict = {
    #         "xs": xs_clean,
    #         "y_trues": y_trues_clean,
    #         "y_preds": y_preds,
    #         "noises": noises,
    #         "y_preds_adversarial": y_preds_adversarial
    #     }
    #     pickle.dump(adv_data_dict, f)
    # print(y_pred)

    # test adversarial examples in MNIST training dataset
    mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    train_dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    num_correct = 0
    for i, data in enumerate(train_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        # start = time.clock()
        perturbation = pretrained_G(test_img)
        perturbation = torch.clamp(perturbation, -0.3, 0.3)
        adv_img = perturbation + test_img
        # elaplsed=time.clock()-start
        # print("Time Used:",elaplsed)
        adv_img = torch.clamp(adv_img, 0, 1)
        pred_lab = torch.argmax(target_model(adv_img.data),-1)
        # pred_lab = torch.argmax(target_model(adv_img), 1)
        num_correct += torch.sum(pred_lab==test_label,0)
        # print(select_tensor(pred_lab, test_label))
        num_success += select_tensor(pred_lab, test_label)
    print('Attack successfully:',num_success)
    print('MNIST training dataset:')
    print('num_correct: ', num_correct.item())
    print('accuracy of adv imgs in training set: %f\n'%(num_correct.item()/len(mnist_dataset)))

    # test adversarial examples in MNIST testing dataset
    mnist_dataset_test = torchvision.datasets.MNIST('./dataset', train=False, transform=transforms.ToTensor(), download=True)
    test_dataloader = DataLoader(mnist_dataset_test, batch_size=batch_size, shuffle=True, num_workers=1)
    num_correct = 0
    num_success = 0
    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        
        init_pred = target_model(test_img).max(1, keepdim = True)[1][0]
        perturbation = pretrained_G(test_img)
        perturbation = torch.clamp(perturbation, -0.3, 0.3)
        adv_img = perturbation + test_img
        adv_img = torch.clamp(adv_img, 0, 1)
        final_pred = target_model(adv_img.data).max(1, keepdim = True)[1][0]
        pred_lab = torch.argmax(target_model(adv_img.data),-1)
        # pred_lab = torch.argmax(target_model(adv_img), 1)
        num_correct += torch.sum(pred_lab==test_label,0)
        num_success += select_tensor(pred_lab, test_label)
        if len(adv_examples) < 5:
            orig_img = test_img[0].squeeze().detach().cpu().numpy()
            adv_ex = adv_img[0].squeeze().detach().cpu().numpy()
            disp_img = np.concatenate((orig_img, adv_ex), axis=1)
            # print('init pred', init_pred)
            # print(adv_ex.shape,adv_img.shape)
            adv_examples.append((init_pred.item(), final_pred.item(), disp_img))
            
    cnt = 0
    for i in range(len(adv_examples)):
        cnt+=1
        plt.subplot(1, len(adv_examples), cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        orig, adv, ex = adv_examples[i]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()
    print('Attack successfully:', num_success)
    print('num_correct: ', num_correct.item())
    print('accuracy of adv imgs in testing set: %f\n'%(num_correct.item()/len(mnist_dataset_test)))

