######################################### CIFAR-10
# e16
# resnet50
CUDA_VISIBLE_DEVICES=0 python -u sample_adv_batch.py -DS cifar10 -C config/adv_cifar10_e16.json -net resnet50 -E 16 


# ######################################### CIFAR-100
# # e16
# # resnet50
# CUDA_VISIBLE_DEVICES=0 python -u sample_adv_batch.py -DS cifar100 -C config/adv_cifar100_e16.json -net resnet50 -E 16 