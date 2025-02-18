# pretreatment of dataset
## 一.预处理文本序列数据集
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
#进行下采样，适当移除部分高频词，频率越高，移除（return false,比较式不成立）概率越大，用random模块增加随机性
#步骤：先分别统计各句子中单词词频，并累积得到总单词数，构建保留词元算法，返回采样后的词元和词频表
#subsample,sentences,vocab,unk,cp,counter,count_corpos,num_tokens，keep,subsampled
def subsample(sentences,vocab):
    #将sentences转化未词元列表，并且排除未知词元<unk>
    sentences = [[token for token in line if vocab[token]!=vocab.unk] for line in sentences]
    counter = cp.count_corpus(sentences)
    #counter是个列表，不能直接累计
    num_tokens = sum(counter.values())
    def keep(token):
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
#逐行获取下采样后的词元，并通过vocab转换成向量形式,最后得到一个张量列表
corpos = [vocab[line]for line in subsampled]
print(corpos[:3])
```
### 3.获取词汇表

### 4.中心词和上下文提取
### 5.随机采样
### 6.负采样（以权重参数为基础）
### 7.加载数据集并构造DataLoader
## 二.预处理数值序列数据(以多特征股票数据为例)
### 1.加载股票数据集
### 2.日期数据类型转换（datetime类型）
### 3.跳过编号和日期列，将数值列转化为numpy数组
### 4.构建样本序列
### 5.自动分区划分数据
### 6.训练集与测试集划分
### 7.将numpy数组转换为tensor,构造DataLoader
