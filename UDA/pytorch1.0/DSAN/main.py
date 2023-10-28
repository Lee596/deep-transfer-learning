import torch
import torch.nn.functional as F
import math
import argparse
import numpy as np
import os
import pandas as pd
import datetime
import wandb

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from DSAN import DSAN
import data_loader


def load_data(root_path, src, tar, batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    loader_src = data_loader.load_training(root_path, src, batch_size, kwargs)
    loader_tar = data_loader.load_training(root_path, tar, batch_size, kwargs)
    loader_tar_test = data_loader.load_testing(
        root_path, tar, batch_size, kwargs)
    return loader_src, loader_tar, loader_tar_test

# 训练一个epoch
def train_epoch(epoch, model, dataloaders, optimizer, df_train_log, batch_idx):
    accu_loss = torch.zeros(1).cuda()
    model.train()
    source_loader, target_train_loader, _ = dataloaders
    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    num_iter = len(source_loader)
    # 遍历每个batch

    for i in range(1, num_iter):
        batch_idx += 1
        data_source, label_source = next(iter_source)
        data_target, _ = next(iter_target)
        if i % len(target_train_loader) == 0:
            iter_target = iter(target_train_loader)
        data_source, label_source = data_source.cuda(), label_source.cuda()
        data_target = data_target.cuda()

        optimizer.zero_grad()
        label_source_pred, loss_lmmd = model(
            data_source, data_target, label_source)
        loss_cls = F.nll_loss(F.log_softmax(
            label_source_pred, dim=1), label_source)
        lambd = 2 / (1 + math.exp(-10 * (epoch) / args.nepoch)) - 1
        loss = loss_cls + args.weight * lambd * loss_lmmd
        accu_loss += loss.detach()

        # 预测标签
        _, preds = torch.max(label_source_pred, dim=1)

        loss.backward()
        optimizer.step()

        # 获取当前 batch 的标签类别和预测类别
        preds = preds.detach().cpu().numpy()
        loss = loss.detach().cpu().numpy()
        label_source = label_source.detach().cpu().numpy()

        log_train = {}
        log_train['epoch'] = epoch
        log_train['batch'] = batch_idx
        # 计算分类评估指标
        log_train['train_loss'] = loss.item()
        log_train['train_accuracy'] = accuracy_score(label_source, preds)
        log_train['train_precision'] = precision_score(label_source, preds, average='macro')
        log_train['train_recall'] = recall_score(label_source, preds, average='macro')
        log_train['train_f1-score'] = f1_score(label_source, preds, average='macro')
        df_train_log = df_train_log.append(log_train, ignore_index=True)
        if i % args.log_interval == 0:
            print(
                f'Epoch: [{epoch:2d}], Loss: {loss.item():.4f}, cls_Loss: {loss_cls.item():.4f}, loss_lmmd: {loss_lmmd.item():.4f}')

    # ======================================================================
    wandb.log({'train_loss': accu_loss.item() / num_iter})
    # ======================================================================
    return df_train_log,batch_idx

def test(model, dataloader):
    model.eval()
    test_loss = 0  # 初始化测试集损失
    correct = 0  # 初始化正确预测的样本数
    loss_list = []
    labels_list = []
    preds_list = []
    with torch.no_grad():
        # 生成一个 batch 的数据和标注
        for data, target in dataloader:
            data, target = data.cuda(), target.cuda()
            pred = model.predict(data)
            # sum up batch loss 计算交叉熵损失并将其添加到测试集损失中
            loss= F.nll_loss(F.log_softmax(pred, dim=1), target).item()
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()
            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            # loss = loss.cpu().numpy()
            pred = pred.cpu().numpy()
            target = target.detach().cpu().numpy()

            loss_list.append(loss)
            labels_list.extend(target)
            preds_list.extend(pred)


        test_loss /= len(dataloader)
        # ======================================================================
        wandb.log({'test_loss': test_loss, 'test_acc': correct / len(dataloader.dataset)})
        # ======================================================================
        print(f'Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(dataloader.dataset)} ({100. * correct / len(dataloader.dataset):.2f}%)')

    log_test = {}
    log_test['epoch'] = epoch

    # 计算分类评估指标
    log_test['test_loss'] = np.mean(loss_list)
    log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
    log_test['test_precision'] = precision_score(labels_list, preds_list, average='macro')
    log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro')
    log_test['test_f1-score'] = f1_score(labels_list, preds_list, average='macro')
    return correct,log_test


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
                        help='batch size', default=36)
    parser.add_argument('--nepoch', type=int,
                        help='Total epoch num', default=5)
    parser.add_argument('--lr', type=list, help='Learning rate', default=[0.001, 0.001, 0.01])
    parser.add_argument('--early_stop', type=int,
                        help='Early stoping number', default=20)
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
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    dataloaders = load_data(args.root_path, args.src,
                            args.tar, args.batch_size)
    model = DSAN(num_classes=args.nclass).cuda()

    correct = 0
    stop = 0

    # 训练日志-测试集
    df_test_log = pd.DataFrame()
    # 训练日志-训练集
    df_train_log = pd.DataFrame()
    batch_idx =0

    if args.bottleneck:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.bottle.parameters(), 'lr': args.lr[1]},
            {'params': model.cls_fc.parameters(), 'lr': args.lr[2]},
        ], lr=args.lr[0], momentum=args.momentum, weight_decay=args.decay)
    else:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.cls_fc.parameters(), 'lr': args.lr[1]},
        ], lr=args.lr[0], momentum=args.momentum, weight_decay=args.decay)

    # ================================================
    '''训练之前，保存日志'''
    os.environ["WANDB_API_KEY"] = "ad04eb473dc9c8f09d7c2956e357f0aafe10e1f4"

    wandb.login()
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    wandb.init(project=args.project_name, config=args.__dict__, name=nowtime, save_code=True)
    model.run_id = wandb.run.id
    # ================================================


    for epoch in range(1, args.nepoch + 1):
        stop += 1
        for index, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = args.lr[index] / math.pow((1 + 10 * (epoch - 1) / args.nepoch), 0.75)


        # 训练
        df_train_log,batch_idx=train_epoch(epoch, model, dataloaders, optimizer,df_train_log ,batch_idx)
        # df_train_log = df_train_log.append(df_train_log, ignore_index=True)
        # 测试
        t_correct,log_test = test(model, dataloaders[-1])
        df_test_log = df_test_log.append(log_test, ignore_index=True)
        if t_correct > correct:
            correct = t_correct
            stop = 0
            torch.save(model, 'model.pkl')
        print(
            f'{args.src}-{args.tar}: max correct: {correct} max accuracy: {100. * correct / len(dataloaders[-1].dataset):.2f}%\n')

        if stop >= args.early_stop:
            print(
                f'Final test acc: {100. * correct / len(dataloaders[-1].dataset):.2f}%')
            break

    # ======================================================================
    wandb.finish()
    # ======================================================================
    # 训练完所有epoch保存日志
    df_train_log.to_csv('训练日志-训练集.csv', index=False)
    df_test_log.to_csv('训练日志-测试集.csv', index=False)
