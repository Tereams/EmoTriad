import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
from tqdm import tqdm
import os
import time
from transformers import BertTokenizer
from transformers import logging

# logging.set_verbosity_error()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class BertSST2Model(nn.Module):

    def __init__(self, class_size, pretrained_name='bert-base-chinese'):
        """
        Args:
            class_size  :指定分类模型的最终类别数目，以确定线性分类器的映射维度
            pretrained_name :用以指定bert的预训练模型
        """
        # 类继承的初始化，固定写法
        super(BertSST2Model, self).__init__()
        # 加载HuggingFace的BertModel
        # BertModel的最终输出维度默认为768
        # return_dict=True 可以使BertModel的输出具有dict属性，即以 bert_output['last_hidden_state'] 方式调用
        self.bert = BertModel.from_pretrained(pretrained_name,
                                              return_dict=True)
        self.speakerLSTM = nn.LSTM(768, 128, bidirectional=True, batch_first=True)
        self.answerLSTM = nn.LSTM(768, 128, bidirectional=True, batch_first=True)
        self.converLSTM = nn.LSTM(768, 128, bidirectional=True, batch_first=True)
        # 通过一个线性层将[CLS]标签对应的维度：768->class_size
        # class_size 在SST-2情感分类任务中设置为：2
        self.classifier = nn.Linear(512, class_size)

    def forward(self, inputs, cut_list):
        # 获取DataLoader中已经处理好的输入数据：
        # input_ids :tensor类型，shape=batch_size*max_len   max_len为当前batch中的最大句长
        # input_tyi :tensor类型，
        # input_attn_mask :tensor类型，因为input_ids中存在大量[Pad]填充，attention mask将pad部分值置为0，让模型只关注非pad部分
        input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs[
            'token_type_ids'], inputs['attention_mask']

        output = self.bert(input_ids, input_tyi, input_attn_mask).pooler_output

        dialogue = []
        count = 0
        for l in cut_list:
            dialogue.append(output[count:count + l])
            count += l

        cover_ans = []
        for d in dialogue:
            conver = self.converLSTM(d)[0]
            cover_ans.append(conver)
        conver_ = torch.cat(cover_ans, 0)

        person_ans = []
        for d in dialogue:
            speaker = self.speakerLSTM(d[::2])[0]
            answer = self.answerLSTM(d[1::2])[0]
            ans = merge_tensor(speaker, answer)
            person_ans.append(ans)
        person_ = torch.cat(person_ans, 0)

        ans = torch.cat([conver_, person_], 1)

        categories_numberic = self.classifier(ans)
        return categories_numberic


def merge_tensor(speaker, answer):
    a1 = speaker.shape[0]
    a2 = answer.shape[0]
    b = speaker.shape[1]
    ans = torch.zeros([a1 + a2, b]).to(speaker.device)
    for i in range(a1):
        ans[2 * i] = speaker[i]
    for i in range(a2):
        ans[2 * i + 1] = answer[i]
    return ans


def save_pretrained(model, path):
    # 保存模型，先利用os模块创建文件夹，后利用torch.save()写入模型文件
    os.makedirs(path, exist_ok=True)
    torch.save(model, os.path.join(path, 'model.pth'))


def readfile(filename, sp=' '):
    f = open(filename, 'r', encoding='utf-8')
    lines = f.readlines()
    ans = []
    for l in lines:
        if sp != ' ':
            ll = l.strip().split(sp)[:-1]
        else:
            ll = l.strip().split(sp)
        ans.append(ll)
    f.close()
    return ans


def load_dialogue(text_path, emo_path, train_ratio=0.8):
    all_data = []

    textss = readfile(text_path, sp='__eou__')
    emoss = readfile(emo_path)

    for z in zip(emoss, textss):
        all_data.append(z)

    length = len(all_data)
    train_len = int(length * train_ratio)
    train_data = all_data[:train_len]
    test_data = all_data[train_len:]

    return train_data, test_data


class BertDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_size = len(dataset)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        data = self.dataset[index]
        ans = []
        for z in zip(data[0], data[1]):
            ans.append(z)
        return ans


def coffate_fn(examples):
    lens = []
    for e in examples:
        lens.append(len(e))
    examples = sum(examples, [])
    inputs, targets = [], []
    for polar, sent in examples:
        inputs.append(sent)
        targets.append(int(polar))
    inputs = tokenizer(inputs,
                       padding=True,
                       truncation=True,
                       return_tensors="pt",
                       max_length=512)
    targets = torch.tensor(targets)
    return inputs, targets, lens


batch_size = 4
num_epoch = 8  # 训练轮次
check_step = 1  # 用以训练中途对模型进行检验：每check_step个epoch进行一次测试和保存模型
text_path = "./final_text.txt"
emo_path = './final_emo.txt'
train_ratio = 0.8  # 训练集比例
learning_rate = 1e-5  # 优化器的学习率


def train(train_data):
    # 将训练数据和测试数据的列表封装成Dataset以供DataLoader加载
    train_dataset = BertDataset(train_data)

    """
    DataLoader主要有以下几个参数：
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load(default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled at every epoch (default: ``False``).
        collate_fn : 传入一个处理数据的回调函数
    DataLoader工作流程：
    1. 先从dataset中取出batch_size个数据
    2. 对每个batch，执行collate_fn传入的函数以改变成为适合模型的输入
    3. 下个epoch取数据前先对当前的数据集进行shuffle，以防模型学会数据的顺序而导致过拟合
    """
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  collate_fn=coffate_fn,
                                  shuffle=False)

    optimizer = Adam(model.parameters(), learning_rate)  # 使用Adam优化器
    CE_loss = nn.CrossEntropyLoss()  # 使用crossentropy作为二分类任务的损失函数

    f1 = open('loss.txt', 'w', encoding='utf-8')
    f2 = open('acc.txt', 'w', encoding='utf-8')

    model.train()
    for epoch in range(1, num_epoch + 1):
        # 记录当前epoch的总loss
        total_loss = 0
        # tqdm用以观察训练进度，在console中会打印出进度条
        for batch in tqdm(train_dataloader, desc=f"Training Epoch: {epoch}"):
            # 因为模型和数据要在同一个设备上才能运行
            inputs, targets = [x.to(device) for x in batch[:-1]]
            cut_list = batch[-1]

            optimizer.zero_grad()
            bert_output = model(inputs, cut_list)

            loss = CE_loss(bert_output, targets)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()
            f1.write(str(loss.item()) + ' ')

        acc = eval(test_data, epoch)
        f2.write(str(acc) + ' ')
    f1.close()
    f2.close()


def eval(test_data, epoch):
    # 记录当前训练时间，用以记录日志和存储
    timestamp = time.strftime("%m_%d_%H_%M", time.localtime())

    test_dataset = BertDataset(test_data)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 collate_fn=coffate_fn)
    # 测试过程
    # acc统计模型在测试数据上分类结果中的正确个数
    acc = 0
    lens = 0
    for batch in tqdm(test_dataloader, desc=f"Testing"):
        inputs, targets = [x.to(device) for x in batch[:-1]]
        cut_list = batch[-1]
        lens += len(targets)
        with torch.no_grad():
            bert_output = model(inputs, cut_list)
            acc += (bert_output.argmax(dim=1) == targets).sum().item()

    # 输出在测试集上的准确率
    print(f"Epoch: {epoch}, Acc: {acc / lens:.4f}")
    return acc / lens

    # if epoch % check_step == 0:
    #     # 保存模型
    #     checkpoints_dirname = "bert_sst2_" + timestamp
    #     os.makedirs(checkpoints_dirname, exist_ok=True)
    #     save_pretrained(model,
    #                     checkpoints_dirname + '/checkpoints-{}/'.format(epoch))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取训练、测试数据、分类类别总数
    train_data, test_data = load_dialogue(text_path=text_path, emo_path=emo_path, train_ratio=train_ratio)

    pretrained_model_name = './pretrained/bert-base-uncased'
    # 创建模型 BertSST2Model
    model = BertSST2Model(3, pretrained_model_name)
    # 固定写法，将模型加载到device上，
    # 如果是GPU上运行，此时可以观察到GPU的显存增加
    model.to(device)
    # 加载预训练模型对应的tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

    train(train_data)
    # eval(test_data,0)
