from torch.autograd import Variable
from torch import nn
import torch
import numpy as np
import logging


def transformer_run_epoch(epoch, model, dataloader, cuda, training=False, optimizer=None):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # inputs = inputs.transpose(1,2)

        if cuda:
            # inputs = torch.from_numpy(inputs)
            # targets = torch.from_numpy(targets)
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        # print('dim:',targets.dim())
        outputs = model(inputs)
        targets = targets.float()
        loss = nn.functional.cross_entropy(outputs, targets)
        if training:
            optimizer.zero_grad() # 清空梯度 用于下一次运算
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct_tmp = 0
        for i in range(0,targets.size(0)):
            num = int(predicted[i])
            if int(targets[i][num]) == 1:
                correct_tmp = correct_tmp + 1
        correct += correct_tmp
    acc = 100 * correct / total
    avg_loss = total_loss / total
    return acc, avg_loss

def to_one_hot(tensor, num_clsses):
    print(tensor.dim())
    assert tensor.dim() <= 1, "[Error] tensor.dim >= 1"
    one_hot = torch.zeros(len(tensor), num_clsses)
    idx = range(len(tensor))
    one_hot[idx, tensor.reshape(-1, )] = 1
    return one_hot

def run_epoch(epoch, model, dataloader, cuda,num_classes=2, training=False, optimizer=None):
    logging.basicConfig(filename='example.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    correct = 0
    total = 0
    total_accuracy = 0
    total_num = 0
    total_pos = 0
    total_neg = 0
    total_classes = np.zeros((num_classes, 1))
    tmp_classes = np.zeros((num_classes, 1))

    for batch_idx, (inputs, targets) in enumerate(dataloader):

        accuracy = 0

        if cuda: inputs, targets = inputs.cuda(), targets.cuda()
        inputs = inputs.permute(0, 2, 1)
        inputs = inputs.float()
        inputs, targets = Variable(inputs), Variable(targets.long())

        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        if training:
            # print(targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)

        total += targets.size(0)
        tmp_pos = 0
        tmp_neg = 0
        for j in range(0, outputs.size(0)):
            out_index = torch.argmax(outputs[j]).item()

            target_index = targets[j].item()


            if out_index == target_index:
                accuracy = accuracy + 1
                total_classes[target_index] = total_classes[target_index] +1
                tmp_classes[target_index] = tmp_classes[target_index] +1
            else:
                total_classes[target_index] = total_classes[target_index] + 1


        total_accuracy = total_accuracy + accuracy
        total_num = total_num + outputs.size(0)

        # print("===========分割线=====================")
        # print('the_classification_is_correct :', total_accuracy)  # 正确分类的个数
        # print("准确率:{}".format(float(total_accuracy / total_num) * 100), '%')

        if cuda:
            correct += predicted.eq(targets.data).cpu().sum().item()
        else:
            correct += predicted.eq(targets.data).sum().item()
    for i in range(0,num_classes):
        rate = tmp_classes[i]/total_classes[i]
        print('第',str(i),'类,准确率:',rate.item())
        # str_tmp = '第{}类,准确率:'.format(rate.item())
        # logging.debug(str_tmp)
        print('总数为:',total_classes[i])
        # str_tmp = '总数为:{}'.format(total_classes[i])
        # logging.debug(str_tmp)
    acc = 100 * correct / total
    avg_loss = total_loss / total
    return acc, avg_loss

def run_epoch_lstm(epoch, model, dataloader, cuda, training=False, optimizer=None):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if cuda: inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets.long())
        # lstm  input = 1000*20*1
        inputs = inputs.transpose(1 ,2)
        inputs = inputs.transpose(0, 1)

        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        for j in range(0, outputs.size(0)):
            out_index = torch.argmax(outputs[j]).item()

            target_index = targets[j].item()

            if out_index == target_index:
                accuracy = accuracy + 1
                if target_index == 0:
                    total_neg = total_neg + 1
                    tmp_neg = tmp_neg + 1
                else:
                    total_pos = total_pos + 1
                    tmp_pos = tmp_pos + 1
            else:
                if target_index == 0:
                    total_neg = total_neg + 1
                else:
                    total_pos = total_pos + 1
            total_right_pos = total_right_pos + tmp_pos
            total_right_neg = total_right_neg + tmp_neg
            total_accuracy = total_accuracy + accuracy
            total_num = total_num + outputs.size(0)

        print("===========分割线=====================")
        print('the_classification_is_correct :', total_accuracy)  # 正确分类的个数
        print("准确率:{}".format(float(total_accuracy / total_num) * 100), '%')
        if cuda:
            correct += predicted.eq(targets.data).cpu().sum().item()
        else:
            correct += predicted.eq(targets.data).sum().item()
    acc = 100 * correct / total
    avg_loss = total_loss / total
    return acc, avg_loss



def run_epoch_cnnlstm(epoch, model, dataloader, cuda, training=False, optimizer=None):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if cuda: inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets.long())

        outputs = model(inputs)
        loss = nn.BCEWithLogitsLossE(outputs, targets)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        if cuda:
            correct += predicted.eq(targets.data).cpu().sum().item()
        else:
            correct += predicted.eq(targets.data).sum().item()
    acc = 100 * correct / total
    avg_loss = total_loss / total
    return acc, avg_loss


def get_predictions(model, dataloader, cuda, get_probs=False): # 可以输出概率
    preds = []
    model.eval()
    # inputs = 10×1×1000
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if cuda: inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets.long())
        outputs = model(inputs)
        if get_probs:
            probs = torch.nn.functional.softmax(outputs, dim=1) # 转换成概率
            if cuda: probs = probs.data.cpu().numpy()
            else: probs = probs.data.numpy()
            preds.append(probs)
        else:
            _, predicted = torch.max(outputs.data, 1) # torch.max(a,1) 返回每一行中最大值的那个元素，且返回其索引
            if cuda: predicted = predicted.cpu()
            preds += list(predicted.numpy().ravel())
    if get_probs:
        return np.vstack(preds)
    else:
        return np.array(preds)