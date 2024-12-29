
import sys

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_network(args):
    """ return given network
    """

    if args.dataset == "cifar10":
        if args.net == 'vgg16':
            from models.net.cifar10.vgg import vgg16_bn
            net = vgg16_bn()
        elif args.net == 'vgg13':
            from models.net.cifar10.vgg import vgg13_bn
            net = vgg13_bn()
        elif args.net == 'vgg11':
            from models.net.cifar10.vgg import vgg11_bn
            net = vgg11_bn()
        elif args.net == 'vgg19':
            from models.net.cifar10.vgg import vgg19_bn
            net = vgg19_bn()
        elif args.net == 'densenet121':
            from models.net.cifar10.densenet import densenet121
            net = densenet121()
        elif args.net == 'densenet161':
            from models.net.cifar10.densenet import densenet161
            net = densenet161()
        elif args.net == 'densenet169':
            from models.net.cifar10.densenet import densenet169
            net = densenet169()
        elif args.net == 'densenet201':
            from models.net.cifar10.densenet import densenet201
            net = densenet201()
        elif args.net == 'googlenet':
            from models.net.cifar10.googlenet import googlenet
            net = googlenet()
        elif args.net == 'inceptionv3':
            from models.net.cifar10.inceptionv3 import inceptionv3
            net = inceptionv3()
        elif args.net == 'inceptionv4':
            from models.net.cifar10.inceptionv4 import inceptionv4
            net = inceptionv4()
        elif args.net == 'inceptionresnetv2':
            from models.net.cifar10.inceptionv4 import inception_resnet_v2
            net = inception_resnet_v2()
        elif args.net == 'xception':
            from models.net.cifar10.xception import xception
            net = xception()
        elif args.net == 'resnet18':
            from models.net.cifar10.resnet import resnet18
            net = resnet18()
        elif args.net == 'resnet34':
            from models.net.cifar10.resnet import resnet34
            net = resnet34()
        elif args.net == 'resnet50':
            from models.net.cifar10.resnet import resnet50
            net = resnet50()
        elif args.net == 'resnet101':
            from models.net.cifar10.resnet import resnet101
            net = resnet101()
        elif args.net == 'resnet152':
            from models.net.cifar10.resnet import resnet152
            net = resnet152()
        elif args.net == 'preactresnet18':
            from models.net.cifar10.preactresnet import preactresnet18
            net = preactresnet18()
        elif args.net == 'preactresnet34':
            from models.net.cifar10.preactresnet import preactresnet34
            net = preactresnet34()
        elif args.net == 'preactresnet50':
            from models.net.cifar10.preactresnet import preactresnet50
            net = preactresnet50()
        elif args.net == 'preactresnet101':
            from models.net.cifar10.preactresnet import preactresnet101
            net = preactresnet101()
        elif args.net == 'preactresnet152':
            from models.net.cifar10.preactresnet import preactresnet152
            net = preactresnet152()
        elif args.net == 'resnext50':
            from models.net.cifar10.resnext import resnext50
            net = resnext50()
        elif args.net == 'resnext101':
            from models.net.cifar10.resnext import resnext101
            net = resnext101()
        elif args.net == 'resnext152':
            from models.net.cifar10.resnext import resnext152
            net = resnext152()
        elif args.net == 'shufflenet':
            from models.net.cifar10.shufflenet import shufflenet
            net = shufflenet()
        elif args.net == 'shufflenetv2':
            from models.net.cifar10.shufflenetv2 import shufflenetv2
            net = shufflenetv2()
        elif args.net == 'squeezenet':
            from models.net.cifar10.squeezenet import squeezenet
            net = squeezenet()
        elif args.net == 'mobilenet':
            from models.net.cifar10.mobilenet import mobilenet
            net = mobilenet()
        elif args.net == 'mobilenetv2':
            from models.net.cifar10.mobilenetv2 import mobilenetv2
            net = mobilenetv2()
        elif args.net == 'nasnet':
            from models.net.cifar10.nasnet import nasnet
            net = nasnet()
        elif args.net == 'attention56':
            from models.net.cifar10.attention import attention56
            net = attention56()
        elif args.net == 'attention92':
            from models.net.cifar10.attention import attention92
            net = attention92()
        elif args.net == 'seresnet18':
            from models.net.cifar10.senet import seresnet18
            net = seresnet18()
        elif args.net == 'seresnet34':
            from models.net.cifar10.senet import seresnet34
            net = seresnet34()
        elif args.net == 'seresnet50':
            from models.net.cifar10.senet import seresnet50
            net = seresnet50()
        elif args.net == 'seresnet101':
            from models.net.cifar10.senet import seresnet101
            net = seresnet101()
        elif args.net == 'seresnet152':
            from models.net.cifar10.senet import seresnet152
            net = seresnet152()
        elif args.net == 'wideresnet':
            from models.net.cifar10.wideresidual import wideresnet
            net = wideresnet()
        elif args.net == 'stochasticdepth18':
            from models.net.cifar10.stochasticdepth import \
                stochastic_depth_resnet18
            net = stochastic_depth_resnet18()
        elif args.net == 'stochasticdepth34':
            from models.net.cifar10.stochasticdepth import \
                stochastic_depth_resnet34
            net = stochastic_depth_resnet34()
        elif args.net == 'stochasticdepth50':
            from models.net.cifar10.stochasticdepth import \
                stochastic_depth_resnet50
            net = stochastic_depth_resnet50()
        elif args.net == 'stochasticdepth101':
            from models.net.cifar10.stochasticdepth import \
                stochastic_depth_resnet101
            net = stochastic_depth_resnet101()

        else:
            print('the network name you have entered is not supported yet')
            sys.exit()
    elif args.dataset == "cifar100":
        if args.net == 'vgg16':
            from models.net.cifar100.vgg import vgg16_bn
            net = vgg16_bn()
        elif args.net == 'vgg13':
            from models.net.cifar100.vgg import vgg13_bn
            net = vgg13_bn()
        elif args.net == 'vgg11':
            from models.net.cifar100.vgg import vgg11_bn
            net = vgg11_bn()
        elif args.net == 'vgg19':
            from models.net.cifar100.vgg import vgg19_bn
            net = vgg19_bn()
        elif args.net == 'densenet121':
            from models.net.cifar100.densenet import densenet121
            net = densenet121()
        elif args.net == 'densenet161':
            from models.net.cifar100.densenet import densenet161
            net = densenet161()
        elif args.net == 'densenet169':
            from models.net.cifar100.densenet import densenet169
            net = densenet169()
        elif args.net == 'densenet201':
            from models.net.cifar100.densenet import densenet201
            net = densenet201()
        elif args.net == 'googlenet':
            from models.net.cifar100.googlenet import googlenet
            net = googlenet()
        elif args.net == 'inceptionv3':
            from models.net.cifar100.inceptionv3 import inceptionv3
            net = inceptionv3()
        elif args.net == 'inceptionv4':
            from models.net.cifar100.inceptionv4 import inceptionv4
            net = inceptionv4()
        elif args.net == 'inceptionresnetv2':
            from models.net.cifar100.inceptionv4 import inception_resnet_v2
            net = inception_resnet_v2()
        elif args.net == 'xception':
            from models.net.cifar100.xception import xception
            net = xception()
        elif args.net == 'resnet18':
            from models.net.cifar100.resnet import resnet18
            net = resnet18()
        elif args.net == 'resnet34':
            from models.net.cifar100.resnet import resnet34
            net = resnet34()
        elif args.net == 'resnet50':
            from models.net.cifar100.resnet import resnet50
            net = resnet50()
        elif args.net == 'resnet101':
            from models.net.cifar100.resnet import resnet101
            net = resnet101()
        elif args.net == 'resnet152':
            from models.net.cifar100.resnet import resnet152
            net = resnet152()
        elif args.net == 'preactresnet18':
            from models.net.cifar100.preactresnet import preactresnet18
            net = preactresnet18()
        elif args.net == 'preactresnet34':
            from models.net.cifar100.preactresnet import preactresnet34
            net = preactresnet34()
        elif args.net == 'preactresnet50':
            from models.net.cifar100.preactresnet import preactresnet50
            net = preactresnet50()
        elif args.net == 'preactresnet101':
            from models.net.cifar100.preactresnet import preactresnet101
            net = preactresnet101()
        elif args.net == 'preactresnet152':
            from models.net.cifar100.preactresnet import preactresnet152
            net = preactresnet152()
        elif args.net == 'resnext50':
            from models.net.cifar100.resnext import resnext50
            net = resnext50()
        elif args.net == 'resnext101':
            from models.net.cifar100.resnext import resnext101
            net = resnext101()
        elif args.net == 'resnext152':
            from models.net.cifar100.resnext import resnext152
            net = resnext152()
        elif args.net == 'shufflenet':
            from models.net.cifar100.shufflenet import shufflenet
            net = shufflenet()
        elif args.net == 'shufflenetv2':
            from models.net.cifar100.shufflenetv2 import shufflenetv2
            net = shufflenetv2()
        elif args.net == 'squeezenet':
            from models.net.cifar100.squeezenet import squeezenet
            net = squeezenet()
        elif args.net == 'mobilenet':
            from models.net.cifar100.mobilenet import mobilenet
            net = mobilenet()
        elif args.net == 'mobilenetv2':
            from models.net.cifar100.mobilenetv2 import mobilenetv2
            net = mobilenetv2()
        elif args.net == 'nasnet':
            from models.net.cifar100.nasnet import nasnet
            net = nasnet()
        elif args.net == 'attention56':
            from models.net.cifar100.attention import attention56
            net = attention56()
        elif args.net == 'attention92':
            from models.net.cifar100.attention import attention92
            net = attention92()
        elif args.net == 'seresnet18':
            from models.net.cifar100.senet import seresnet18
            net = seresnet18()
        elif args.net == 'seresnet34':
            from models.net.cifar100.senet import seresnet34
            net = seresnet34()
        elif args.net == 'seresnet50':
            from models.net.cifar100.senet import seresnet50
            net = seresnet50()
        elif args.net == 'seresnet101':
            from models.net.cifar100.senet import seresnet101
            net = seresnet101()
        elif args.net == 'seresnet152':
            from models.net.cifar100.senet import seresnet152
            net = seresnet152()
        elif args.net == 'wideresnet':
            from models.net.cifar100.wideresidual import wideresnet
            net = wideresnet()
        elif args.net == 'stochasticdepth18':
            from models.net.cifar100.stochasticdepth import \
                stochastic_depth_resnet18
            net = stochastic_depth_resnet18()
        elif args.net == 'stochasticdepth34':
            from models.net.cifar100.stochasticdepth import \
                stochastic_depth_resnet34
            net = stochastic_depth_resnet34()
        elif args.net == 'stochasticdepth50':
            from models.net.cifar100.stochasticdepth import \
                stochastic_depth_resnet50
            net = stochastic_depth_resnet50()
        elif args.net == 'stochasticdepth101':
            from models.net.cifar100.stochasticdepth import \
                stochastic_depth_resnet101
            net = stochastic_depth_resnet101()

        else:
            print('the network name you have entered is not supported yet')
            sys.exit()
    elif args.dataset == "tinyimagenet":
        if args.net == 'vgg16':
            from models.net.tinyimagenet.vgg import vgg16_bn
            net = vgg16_bn()
        elif args.net == 'vgg13':
            from models.net.tinyimagenet.vgg import vgg13_bn
            net = vgg13_bn()
        elif args.net == 'vgg11':
            from models.net.tinyimagenet.vgg import vgg11_bn
            net = vgg11_bn()
        elif args.net == 'vgg19':
            from models.net.tinyimagenet.vgg import vgg19_bn
            net = vgg19_bn()
        elif args.net == 'densenet121':
            from models.net.tinyimagenet.densenet import densenet121
            net = densenet121()
        elif args.net == 'densenet161':
            from models.net.tinyimagenet.densenet import densenet161
            net = densenet161()
        elif args.net == 'densenet169':
            from models.net.tinyimagenet.densenet import densenet169
            net = densenet169()
        elif args.net == 'densenet201':
            from models.net.tinyimagenet.densenet import densenet201
            net = densenet201()
        elif args.net == 'googlenet':
            from models.net.tinyimagenet.googlenet import googlenet
            net = googlenet()
        elif args.net == 'inceptionv3':
            from models.net.tinyimagenet.inceptionv3 import inceptionv3
            net = inceptionv3()
        elif args.net == 'inceptionv4':
            from models.net.tinyimagenet.inceptionv4 import inceptionv4
            net = inceptionv4()
        elif args.net == 'inceptionresnetv2':
            from models.net.tinyimagenet.inceptionv4 import inception_resnet_v2
            net = inception_resnet_v2()
        elif args.net == 'xception':
            from models.net.tinyimagenet.xception import xception
            net = xception()
        elif args.net == 'resnet18':
            from models.net.tinyimagenet.resnet import resnet18
            net = resnet18()
        elif args.net == 'resnet34':
            from models.net.tinyimagenet.resnet import resnet34
            net = resnet34()
        elif args.net == 'resnet50':
            from models.net.tinyimagenet.resnet import resnet50
            net = resnet50()
        elif args.net == 'resnet101':
            from models.net.tinyimagenet.resnet import resnet101
            net = resnet101()
        elif args.net == 'resnet152':
            from models.net.tinyimagenet.resnet import resnet152
            net = resnet152()
        elif args.net == 'preactresnet18':
            from models.net.tinyimagenet.preactresnet import preactresnet18
            net = preactresnet18()
        elif args.net == 'preactresnet34':
            from models.net.tinyimagenet.preactresnet import preactresnet34
            net = preactresnet34()
        elif args.net == 'preactresnet50':
            from models.net.tinyimagenet.preactresnet import preactresnet50
            net = preactresnet50()
        elif args.net == 'preactresnet101':
            from models.net.tinyimagenet.preactresnet import preactresnet101
            net = preactresnet101()
        elif args.net == 'preactresnet152':
            from models.net.tinyimagenet.preactresnet import preactresnet152
            net = preactresnet152()
        elif args.net == 'resnext50':
            from models.net.tinyimagenet.resnext import resnext50
            net = resnext50()
        elif args.net == 'resnext101':
            from models.net.tinyimagenet.resnext import resnext101
            net = resnext101()
        elif args.net == 'resnext152':
            from models.net.tinyimagenet.resnext import resnext152
            net = resnext152()
        elif args.net == 'shufflenet':
            from models.net.tinyimagenet.shufflenet import shufflenet
            net = shufflenet()
        elif args.net == 'shufflenetv2':
            from models.net.tinyimagenet.shufflenetv2 import shufflenetv2
            net = shufflenetv2()
        elif args.net == 'squeezenet':
            from models.net.tinyimagenet.squeezenet import squeezenet
            net = squeezenet()
        elif args.net == 'mobilenet':
            from models.net.tinyimagenet.mobilenet import mobilenet
            net = mobilenet()
        elif args.net == 'mobilenetv2':
            from models.net.tinyimagenet.mobilenetv2 import mobilenetv2
            net = mobilenetv2()
        elif args.net == 'nasnet':
            from models.net.tinyimagenet.nasnet import nasnet
            net = nasnet()
        elif args.net == 'attention56':
            from models.net.tinyimagenet.attention import attention56
            net = attention56()
        elif args.net == 'attention92':
            from models.net.tinyimagenet.attention import attention92
            net = attention92()
        elif args.net == 'seresnet18':
            from models.net.tinyimagenet.senet import seresnet18
            net = seresnet18()
        elif args.net == 'seresnet34':
            from models.net.tinyimagenet.senet import seresnet34
            net = seresnet34()
        elif args.net == 'seresnet50':
            from models.net.tinyimagenet.senet import seresnet50
            net = seresnet50()
        elif args.net == 'seresnet101':
            from models.net.tinyimagenet.senet import seresnet101
            net = seresnet101()
        elif args.net == 'seresnet152':
            from models.net.tinyimagenet.senet import seresnet152
            net = seresnet152()
        elif args.net == 'wideresnet':
            from models.net.tinyimagenet.wideresidual import wideresnet
            net = wideresnet()
        elif args.net == 'stochasticdepth18':
            from models.net.tinyimagenet.stochasticdepth import \
                stochastic_depth_resnet18
            net = stochastic_depth_resnet18()
        elif args.net == 'stochasticdepth34':
            from models.net.tinyimagenet.stochasticdepth import \
                stochastic_depth_resnet34
            net = stochastic_depth_resnet34()
        elif args.net == 'stochasticdepth50':
            from models.net.tinyimagenet.stochasticdepth import \
                stochastic_depth_resnet50
            net = stochastic_depth_resnet50()
        elif args.net == 'stochasticdepth101':
            from models.net.tinyimagenet.stochasticdepth import \
                stochastic_depth_resnet101
            net = stochastic_depth_resnet101()

        else:
            print('the network name you have entered is not supported yet')
            sys.exit()
    else:
        print('the dataset you have entered is not supported yet')
        sys.exit()

    return net
