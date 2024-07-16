
import argparse
from tqdm import tqdm
import core.praser as Praser
import torch
from models.network import Network
from util import get_network,get_data
from torchvision.utils import save_image
import numpy as np
import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def denorm(x):
    return (x+1)/2


def sample_adv(model, cond_image, eps):

    output, _ = model.restoration(cond_image, sample_num=100)

    output = denorm(output)
    cond_image = denorm(cond_image)
    adv_noise = output - cond_image
    adv_noise = torch.clamp(adv_noise, -eps/255, eps/255)
    adv_image = cond_image+adv_noise
    adv_image = torch.clamp(adv_image, 0, 1)

    return output, adv_image


def main(args):
    # config arg

    opt = Praser.parse(args)

    model_args = opt["model"]["which_networks"][0]["args"]
    model_pth = opt["path"]["resume_state"]

    # initializa model
    model = Network(**model_args)
    state_dict = torch.load(model_pth)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.set_new_noise_schedule(phase='test')
    model.eval()


    trainloader, testloader = get_data(args)

    net = get_network(args)

  
    pth_path = F"checkpoint/victims/{args.dataset}/{args.net}.pth"
    net.load_state_dict(torch.load(pth_path))
    net.to(device)
    net.eval()


    total_img_num = 0
    success = 0
    queries = []

    with torch.no_grad():
        for i_batch, (x, y) in enumerate(tqdm(testloader, disable=True)):
            cond_image = x.to(device)
            y = y.to(device)
            target_label = torch.zeros_like(y)

            pred = net(denorm(cond_image)).argmax(1)
            # print(pred)
            correct_1 = pred.eq(y)
            correct_2 = pred.ne(target_label)
            correct = correct_1 & correct_2

            index_correct = torch.nonzero(correct).squeeze(1)

            cond_image = cond_image[index_correct]
            y = y[index_correct]
            # print(pred,"\n",y)
            target_label = target_label[index_correct]

            total_img_num += index_correct.size(0)

            for query in range(1, 1+args.max_queries):
                starttime = datetime.datetime.now()
                output, adv_image = sample_adv(model, cond_image, args.eps)
          
                pred_adv = net(adv_image).argmax(1)

                un_succ = pred_adv.ne(target_label)
                
                index_un_succ = torch.nonzero(un_succ).squeeze(1)

                succ = pred_adv.eq(target_label)
                success += succ.sum().item()
                endtime = datetime.datetime.now()


                log_msg = F"Batch: {i_batch}\tQuery: {query}\tSuccessed: {success}\tRemain: {correct.sum()-success}\tSuccess_Rate: {success*100/total_img_num:.4f}\tTime: {(endtime - starttime).seconds}s"

                print(log_msg)

                queries = queries+[query]*succ.sum()

                if index_un_succ.size(0) == 0:
                    break

                cond_image = cond_image[index_un_succ]
                y = y[index_un_succ]
                target_label = target_label[index_un_succ]

            if (i_batch+1)*args.batch_size==args.attack_num:
                break
            

        print("\n\n\n")

        print(F"Queries:\t", queries)

        print(F"Total Attacked:\t {total_img_num}")
        print(F"Attack Success Rate:\t {success*100/total_img_num:.4f}")
        print(F"Classify Success Rate:\t", total_img_num *
              100/args.batch_size/(i_batch+1))
        print("Average Queries:\t{:.4f} ".format(
            sum(queries)/int(success)))
        print("Median Query:\t{:.4f}".format(np.median(queries)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')

    parser.add_argument('-C', '--config', type=str,
                        default='config/target/adv_cifar10_e16.json', help='JSON file for configuration')
    parser.add_argument('-P', '--phase', type=str,
                        choices=['train', 'test'], help='Run train or test', default='test')
    parser.add_argument('-B', '--batch', type=int,
                        default=None, help='Batch size in every gpu')  # 不能删
    # parser.add_argument('-bs', "--batch_size", type=int, default=1000,
    #                     help='batch size for dataloader')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)  # 不能删
    parser.add_argument('-d', '--debug', action='store_true')  # 不能删
    parser.add_argument('-E', '--eps', default=16, type=int)
    parser.add_argument('-AN', '--attack_num', default=10000, type=int)
    parser.add_argument('-Q', '--max_queries', default=1000, type=int)
    parser.add_argument('-DS', '--dataset', type=str, default="cifar10", choices=['cifar10','svhn','stl10', 'cifar100', 'tinyimagenet'],
                        help='dataset to use')

    args = parser.parse_args()

    args.batch_size = 1000
    print(args)
    main(args)
