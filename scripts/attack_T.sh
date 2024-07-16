######################################### CIFAR-10
# E16
# resnet50
CUDA_VISIBLE_DEVICES=1 python -u sample_adv_batch_T.py -DS cifar10 -C config/adv_cifar10_e16_T.json -net resnet50 -E 16


# ######################################### CIFAR-100
# # E16
# # resnet50
# CUDA_VISIBLE_DEVICES=1 python -u sample_adv_batch_T.py -DS cifar100 -C config/adv_cifar100_e16_T.json -net resnet50 -E 16