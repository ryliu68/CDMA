import torch


adv_imgs = torch.rand(100, 3, 32, 32)
imgs = torch.rand(100, 3, 32, 32)


data = {
    "adv_imgs": adv_imgs,
    "imgs": imgs,
}


torch.save(data, "data/adv/cifar10/FAKE_train_eps_16.pth")
