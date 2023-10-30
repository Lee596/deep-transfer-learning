import torch
import torch.nn.functional as F
import math
import argparse
import numpy as np
import os
import pandas as pd
import data_loader

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

def load_data(root_path, src, tar, batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    loader_src = data_loader.load_training(root_path, src, batch_size, kwargs)
    loader_tar = data_loader.load_training(root_path, tar, batch_size, kwargs)
    loader_tar_test = data_loader.load_testing(
        root_path, tar, batch_size, kwargs)
    return loader_src, loader_tar, loader_tar_test

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    loss_list = []
    labels_list = []
    preds_list = []
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.cuda(), target.cuda()
            pred =  model.predict(data)
            loss = F.nll_loss(F.log_softmax(pred, dim=1), target).item()
            pred = pred.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            total += data.size(0)

            loss_list.append(loss)
            labels_list.extend(target.detach().cpu().numpy())
            preds_list.extend(pred.detach().cpu().numpy())

    accuracy = 100. * correct / total
    precision = precision_score(labels_list, preds_list, average='macro')
    recall = recall_score(labels_list, preds_list, average='macro')
    f1 = f1_score(labels_list, preds_list, average='macro')
    test_loss = np.mean(loss_list)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, total, accuracy))

    print('Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}'.format(
        precision, recall, f1))

    return accuracy, precision, recall, f1, test_loss

def get_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, help='Root path for dataset',
                        default='indoor-field')
    parser.add_argument('--root_path', type=str, help='Root path for dataset',
                        default='../../../data/ricev1')
    parser.add_argument('--src', type=str,
                        help='Source domain', default='indoor')
    parser.add_argument('--tar', type=str,
                        help='Target domain', default='field2')
    parser.add_argument('--nclass', type=int,
                        help='Number of classes', default=4)
    parser.add_argument('--batch_size', type=float,
                        help='batch size', default=32)
    parser.add_argument('--nepoch', type=int,
                        help='Total epoch num', default=200)
    parser.add_argument('--lr', type=list, help='Learning rate', default=[0.001, 0.001, 0.01])
    parser.add_argument('--early_stop', type=int,
                        help='Early stoping number', default=30)
    parser.add_argument('--seed', type=int,
                        help='Seed', default=2021)
    parser.add_argument('--weight', type=float,
                        help='Weight for adaptation loss', default=0.5)
    parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
    parser.add_argument('--decay', type=float,
                        help='L2 weight decay', default=5e-4)
    parser.add_argument('--bottleneck', type=str2bool,
                        nargs='?', const=True, default=True)
    parser.add_argument('--log_interval', type=int,
                        help='Log interval', default=10)
    parser.add_argument('--gpu', type=str,
                        help='GPU ID', default='0')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print(vars(args))
    # SEED = args.seed
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # 有 GPU 就用 GPU，没有就用 CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device', device)

    dataloaders = load_data(args.root_path, args.src,
                            args.tar, args.batch_size)
    a = dataloaders[-1]
    print(dataloaders[-1])
    '''导入训练好的模型'''
    from DSAN import DSAN
    model = torch.load('./model.pkl', map_location=torch.device('cpu'))
    model.to(device)

    from torchvision import transforms

    # # 训练集图像预处理：缩放裁剪、图像增强、转 Tensor、归一化
    # train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
    #                                       transforms.RandomHorizontalFlip(),
    #                                       transforms.ToTensor(),
    #                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #                                      ])

    # 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                                         ])


    '''载入测试集'''
    # 数据集文件夹路径
    dataset_dir = '../../../data/ricev1/'
    test_path = os.path.join(dataset_dir, 'field2')
    from torchvision import datasets

    # 载入测试集
    test_dataset = datasets.ImageFolder(test_path, test_transform)
    print('测试集图像数量', len(test_dataset))
    print('类别个数', len(test_dataset.classes))
    print('各类别名称', test_dataset.classes)
    evaluate(model, dataloaders[-1])
