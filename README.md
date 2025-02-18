# pretreatment of dataset
## 一.预处理文本序列数据集
提前载入预训练过程中需要用到的库，其中matplotlib主要用于可视化
```python
import torch
from torch import nn
from d2l import torch as d2l
import pandas as pd
import math
import matplotlib.pyplot as plot
import random
import corpos_vocab_package as cp
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'
```
### 1.下载数据集并返回文件名
本质上就是通过数据集所在url进行链接，将文件下载到文件夹，并返回文件路径
```python
#传入参数包括url，默认文件夹路径，以及SHA-1哈希校验值，提供后可以与本地文件的哈希值进行比较检验，避免文件重复
def download(url, folder='../data', sha1_hash=None):
    #针对不是http开头的url（不完整）,会从DATAHUB[]字典(dict)中获取当前url完整的url以及对应的哈希检验值，并更新
    if not url.startswith('http'):
        #便于向后兼容
        url, sha1_hash = DATA_HUB[url]
    #在当前路径创建文件夹，并且将创建的文件夹路径和文件名（url.split('/')[-1]）进行拼接（join）来得到文件路径
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, url.split('/')[-1])
    # 判断本地是否存在该文件
    if os.path.exists(fname) and sha1_hash:
        #创建hashlib.sha1()对象，即sha1,用于在之后读取数据块时，根据数据块进行计算哈希值
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                #每次读取1MB数据，即1048576字节的数据，进行更新
                data = f.read(1048576)
                if not data:
                    break
                #每次读取data就计算并更新哈希值
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    #如果本地不存在文件，就会通过request方法，通过得到的url,利用流式传输，验证(verify)SSL证书后进行下载，然后将内容写入文件夹中，最后返回文件路径
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname
#下载并解压文件
def download_extract(name, folder=None):
    fname = download(name)
    #获取文件路径，并通过splitext，将文件路径拆分成文件名和文件后缀，然后根据后缀类型，选择对应的文件解压方式进行解压
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    #将压缩文件夹中的所有内容解压到base_dir下
    fp.extractall(base_dir)
    #如果传入参数提供了文件保存的子目录名folder,就将base_dir和folder直接进行链接，如果不是，就将
    return os.path.join(base_dir, folder) if folder else data_dir

#将ptb数据集的下载链接以及其校验和（哈希值）加入DATA_HUB中,然后再解压并获取文件路径，打开文件，并将文件中的字符序列按行分割，并分解为一个个的单词，并以列表的形式返回
#read_ptb,data_dir,ptb.train.txt,f,raw_text,
#@save
d2l.DATA_HUB['ptb'] = (d2l.DATA_URL+'ptb.zip','319d85e578af0cdc590547f26231e4e31cdf1e42')
#@save
def read_ptb():
    data_dir = d2l.download_extract('ptb')
    with open(os.path.join(data_dir,'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]
```
### 2.下采样
下采样是针对于文本序列中大量价值较低的高频词进行设计的采样方式，根据词频适当移除部分高频词
```python
#进行下采样，适当移除部分高频词，频率越高，counter[token]越大，移除（return false,比较式不成立）概率越大，用random模块增加随机性
#步骤：先分别统计各句子中单词词频，并累积得到总单词数，构建保留词元算法，返回采样后的词元和词频表
#subsample,sentences,vocab,unk,cp,counter,count_corpos,num_tokens，keep,subsampled
def subsample(sentences,vocab):
    #将sentences转化未词元列表，并且排除未知词元<unk>
    sentences = [[token for token in line if vocab[token]!=vocab.unk] for line in sentences]
    counter = cp.count_corpus(sentences)
    #counter是个列表，不能直接累计
    num_tokens = sum(counter.values())
    def keep(token):
        #此处是下采样函数的核心实现
        return (random.uniform(0,1)<math.sqrt(1e-4/counter[token]*num_tokens))
    return ([[token for token in line if keep(token)]for line in sentences],counter)
subsampled, counter = subsample(sentences,vocab)
#绘制出的图像，横坐标表示词频，经过下采样之后，高词频的单词基本都变成低词频了，总体词频集中在低词频区域
d2l.show_list_len_pair_hist(['origin','subsampled'],'#tokens per sentence','count',sentences,subsampled)
plot.show()

#创建一个进行词元频率先后比较的函数，在词元列表中逐个获取单词，然后进行统计
def compare_counts(token):
    return (f'"{token}"的数量:'
    f'former={sum([l.count(token) for l in sentences])},'
    f'subsampled={sum([l.count(token) for l in subsampled])}')
print(compare_counts('the'))
```
### 3.获取词汇表
```python
#逐行获取下采样后的词元，并通过vocab转换成向量形式,最后得到一个张量列表
corpos = [vocab[line]for line in subsampled]
print(corpos[:3])
```
### 4.中心词和上下文提取
```python
#中心词和上下文提取
#先初始化中心词和上下文词列表，逐行获取词向量表，跳过过短的句子（少于两个词元），逐行加入中心词
#然后将句子中的每一个词逐个作为中心词，通过上下文窗口得到上下文词（跳元模型）,利用上下文索引逐个将加入上下文词
#get_centers_and_contexts,corpos,max_window_size,centers,contexts,line,corpos,,window_size,indice,idx
def get_centers_and_contexts(corpos,max_window_size):
    centers = []
    contexts = []
    for line in corpos:
        if len(line)<2:
            continue
        centers+=line
        #!!!range内部参数应该是整数，而不能是列表，要用len(line),注意内部indices的索引范围，最低不低于0，最大不大于len(line)
        for i in range(len(line)):
           window_size = random.randint(1,max_window_size)
           indices = list(range(max(0,i-window_size),
                                min(len(line), i+1+window_size))) 
           #从上下文词中移除中心词,然后一次也是加入一个中心词对应的上下文词
           indices.remove(i)
           contexts.append([line[idx]for idx in indices])
    return centers,contexts
#进行测试,创建小数据集0-6，7-9进行测试，分别作为centers,contexts,并且将返回值解包后进行逐个读取
#tiny_dataset,center,context,all_centers,all_contexts
tiny_dataset = [list(range(7)),list(range(7,10))]
for center,context in zip(*get_centers_and_contexts(tiny_dataset,2)):
    #print(f'中心词{center}的上下文词是{context}')
    print('中心词',center,'的上下文词是',context)
all_centers ,all_contexts = get_centers_and_contexts(corpos,5)
print(f'"中心词-上下文词对"的数量:{sum([len(context)for context in all_contexts])}')
```
### 5.随机采样
```python
#创建随机采样函数，根据提供的n个采样权重在{1,....,n}的索引中进行随机抽取
#先初始化索引列表polulation，权重列表sampling_weights，以及采样缓存区candidates，缓存区索引i
#@save
#RandomGenerator,self,sampling_weights,candidates,i,draw
class RandomGenerator():
    def __init__(self,sampling_weights):
        self.sampling_weights  = sampling_weights
        self.population = list(range(1,len(sampling_weights)+1))
        self.candidates = []
        self.i = 0
    
    def draw(self):
        #当多次调用draw函数，缓冲区中已经满了，存储了len(self.candidates)个采样结果时，i就需要重新从0开始进行缓存,先缓存k个随机结果来初始化
        if self.i == len(self.candidates):
            #random.choices函数将随机采样权重列表中的参数逐个应用在同样形状的索引列表population,在首次随机采样进行填充过后，candidates长度就变为了k
            self.candidates = random.choices(self.population,self.sampling_weights,k = 10000)
            self.i = 0
        self.i+=1
        #从已经填充好的缓存中返回当前i索引位置的元素并返回调用者
        return self.candidates[self.i-1]
#@save
#generator,_
generator = RandomGenerator([2,3,4])
print([generator.draw() for _ in range(10)])
```
### 6.负采样（以权重参数为基础）
```python
#先根据词汇表中各单词的词频进行初始化权重参数，然后进行随机采样，逐行获取上下文，并将加入并返回len(contexts)*K个噪声词，并保证噪声词不能是上下文词
#@save
#get_negatives,all_contexts,vocab,counter,K,to_tokens,sampling_weights,all_negatives,generator,contexts,negatives,neg
def get_negatives(all_contexts, vocab, counter, K):
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75 for i in range(1,len(vocab))]
    all_negatives,generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        #对于每组上下文词，生成len(contexts) * K个噪声词，利用generator.draw逐个获取噪声词，添加在噪声词组后，再按组添加到all_negatives中
        while len(negatives)<len(contexts) * K:
            neg = generator.draw()
            #噪声词不能是上下文词
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

all_negatives = get_negatives(all_contexts, vocab, counter, 5)
```
### 7.加载数据集并构造DataLoader
```python
#获取带有负采样(negatives)的跳元模型的小批量样本，用于将大小不同的centers,contexts,negatives进行填充
#按照最长的len(centers)+len(contexts)进行样本对齐
#@save
#在构建小批量的函数中传入单个样本为(centers,contexts,negatives)的三元组，通过样本获取三元组中上下文词和噪声词的长度，并取较大值为max_len,用于
#之后进行的掩码构建。然后初始化centers,contexts_negatives,masks,labels,然后从data中逐个获取center,context,negative,并构建上述初始化列表
#掩码masks是针对于经过拼接和填充后的centers_negatives进行构建的，cur_len=len(centers)+len(contexts)作为有效部分（用1），
#剩余max_len-cur_len作为无效部分（用0）,而label将contexts部分进行标记(用1),其余部分都用0（max_len-len(context)）,最后用torch.tensor()转化张量
#batchify, data, max_len, centers, contexts_negatives, masks, labels, center, context, negative, cur_len
def batchify(data):
    #根据传入的data样本中最长的len(contexts)+len(negatives)
    max_len = max(len(c)+len(n) for _, c, n in data )
    centers,contexts_negatives,masks,labels = [],[],[],[]
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        #加[center]用于扩展列表
        centers += [center]
        contexts_negatives += [context+negative +[0] * (max_len-cur_len)]
        #注意加等于
        masks += [cur_len * [1] + (max_len-cur_len) * [0]]
        labels += [len(context) * [1] + (max_len-len(context)) * [0]]
    return (torch.tensor(centers).reshape((-1,1)), 
            torch.tensor(contexts_negatives), 
            torch.tensor(masks), 
            torch.tensor(labels))


#测试
#x_1,x_2, batch, names, name, data
x_1 = (1, [2,2], [3,3,3,3])
x_2 = (1, [2,2,2], [3,3])
#传入单一元组，就可以对每个样本元素(x_1)(x_2)进行单独处理，而使用x_1+x_2就会先进行拼接在进行批量化
batch = batchify((x_1,x_2))
names = ['centers', 'contexts_negatives', 'masks', 'labels']
#最后获取的data是一个包括'centers', 'contexts_negatives', 'masks', 'labels'的四元组
for name, data  in zip(names, batch):
    print(name, '=', data)
```
### 8.汇总上述功能，并通过load_data_ptb()接口进行调用
```python
#统合上述操作，进行加载数据集到内存中，并通过加载器(torch.utils.data.DataLoader)转换为迭代器
#传入dataloader线程数,读取数据集，解压加载进内存中，通过数据集数据创建词汇表，进行下采样并获取词频表，逐行读取下采样后的样本并转换为列表格式，构建词表
#根据词表corpus和最大窗口长度，获取中心词和上下文词，然后进行负采样，获取噪声词
#创建数据集类，用于在torch.util.data.DataLoader中进行加载数据，且便于进行自定义格式调整，需要有__get_item__和__len__方法实现（DataLoader要求）
#数据集在每个批次(batch)中都需要进行打乱顺序(shuffle=True)，并应用collate=batchify(自定义批量化方式，用于将不同大小的centers,contexts,negatives进行填充)
#子线程数num_workers设置为0，不启用多线程，因为pickle序列化会出现问题，无法运行多线程
#load_data_ptb,batch_size,max_window_size,num_noise_words,num_workers,sentences,vocab,subsampled,counter,corpus,all_centers,all_contexts,corpus,max_window_size
#all_negatives,all_contexts,vocab,counter,num_noise_words,PTBataset,__init__,__getitem__,__len__,dataset
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    num_workers = d2l.get_dataloader_workers()
    sentences = read_ptb()
    vocab = cp.Vocab(sentences,min_freq=10)
    subsampled, counter= subsample(sentences,vocab)
    corpus = [vocab[line]for line in subsampled]
    all_centers,all_contexts = get_centers_and_contexts(corpus,max_window_size)
    all_negatives = get_negatives(all_contexts,vocab,counter,num_noise_words)
    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives
        def __getitem__(self, index):
            return (self.centers[index],self.contexts[index],self.negatives[index])
        def __len__(self):
            #由于在初始化函数中将centers,contexts,negatives的长度进行了统一，因此实际上返回任意一个的长度都可以
            return len(self.centers)
    dataset = PTBDataset(all_centers, all_contexts, all_negatives)
    #由于PTBDataset类定义在函数内部，如果使用多线程，在数据集迭代器读取的过程中会出现pickle问题，因此暂时只能禁用多线程，将子线程数量设置为0
    #collate_fn会自动对输入的多个样本(list,tuple)转化为张量(tensor)形式，并进行堆叠
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, collate_fn=batchify, num_workers=0  )
    return data_iter, vocab

#针对上述函数进行测试，确保正确得到对应的数据迭代器data_iter
#因此，最终得到了经过下采样，负采样，以及批量化处理过后的数据迭代器data_iter，内部元素样本从原来的centers, contexts, negatives三元组变为
#centers,contexts_negatives, masks, labels格式的四元组，可以很方便的对contexts_negatives应用masks掩码计算
data_iter, vocab= load_data_ptb(512, 5, 5)
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name,'shape',data.shape)
    break
```
## 二.预处理数值序列数据(以多特征股票数据为例)
```python
import torch
from torch import nn
from d2l import torch as d2l
import pandas as pd
import math
import matplotlib.pyplot as plot
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'
```
### ！预处理数值序列数据背景以及作用分析
常规的进行股票数据处理的模型为RNN模型，将连续多日的数据作为单元进行预测，但是这种方式往往不能充分利用大量股票交易过程中的特征值，包括开盘价，交易量，交易总额，交易高价和交易低价，涨跌幅度等，因此考虑引入transformer模型以及模型中涉及的注意力机制，尝试挖掘各特征值之间的潜在练习，由于完整的transformer模型是seq2seq模型，输入输出都是经过词元转化后的序列数据，这与股票预测(回归类型任务)的目标有一定区别，因此经过多次尝试，最终在GPT模型的启发下，确定使用TransformerDecoder模块，针对各特征值进行归一化后，并在最终得到的预测特征值后通过一个全连接层，将序列数据转换成一个数值数据，从而实现回归预测结果。

通过在互联网上收集数据集，发现了股票数据略有缺失，但是获取途径相对方便的tushare数据集，常见针对tushare数据集进行处理的操作可以参考本作者的stock_analysis repository仓库中的README.md文档
### 1.加载股票数据集
```python

```
### 2.日期数据类型转换（datetime类型）
```python

```
### 3.跳过编号和日期列，将数值列转化为numpy数组
```python

```
### 4.构建样本序列
```python

```
### 5.自动分区划分数据
```python

```
### 6.训练集与测试集划分
```python

```
### 7.将numpy数组转换为tensor,构造DataLoader
```python

```
