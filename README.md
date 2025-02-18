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

def download_extract(name, folder=None):
    """Download and extract a zip/tar file.

    Defined in :numref:`sec_utils`"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
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
