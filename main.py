import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from advGAN import AdvGAN_Attack
from models import MNIST_target_net
from train_mnist import Net


def flat_trans(x):
    x.resize_(28*28)
    return x


if __name__ == '__main__':
    use_cuda=True
    image_nc=1
    epochs = 400                                                                                                                                                                                                                                                                                                                             
    batch_size = 256
    BOX_MIN = 0
    BOX_MAX = 1

    # Define what device we are using
    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    pretrained_model = "./target_model_fcn.pth"
    targeted_model = Net().to(device)
    targeted_model.load_state_dict(torch.load(pretrained_model))
    targeted_model.eval()
    model_num_labels = 10
    # mnist_transform = transforms.Compose(
    #     #     [transforms.ToTensor(), transforms.Lambda(flat_trans)]
    #     # )
    mnist_transform = transforms.ToTensor()
    # MNIST train dataset and dataloader declaration
    mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=mnist_transform, download=True)
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    advGAN = AdvGAN_Attack(device,
                          targeted_model,
                          model_num_labels,
                          image_nc,
                          BOX_MIN,
                          BOX_MAX)

    advGAN.train(dataloader, epochs)
