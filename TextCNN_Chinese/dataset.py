from torchtext import data
import jieba
import logging
from config import Config
jieba.setLogLevel(logging.INFO)


def cut(sentence):
	return [token for token in jieba.lcut(sentence) if token not in stopwords]

def get_iter(config,args):

	stopwords = open(config.stopwords_path).read().split('\n')

	TEXT = torchtext.data.Field(sequential=True,lower=True,tokenize=cut)
	LABEL = torchtext.data.LabelField(sequential=False, dtype=torch.int64)
	train_dataset,dev_dataset,test_dataset = 
	torchtext.data.TabularDataset.splits(
		path=config.datapath,                 #文件存放路径
		format=config.data_format,                  #文件格式
		skip_header=False,             #是否跳过表头，我这里数据集中没有表头，所以不跳过
		train=config.train_name,  
		validation=config.dev_name,
		test=config.test_name,    
		fields=[('label',LABEL),('content',TEXT)] # 定义数据对应的表头
		)
	vectors = torchtext.vocab.Vectors(name=config.pretrained_name, 
		cache=config.pretrained_path)
	
	TEXT.build_vocab(train_dataset, dev_dataset,test_dataset,vectors=vectors)
	LABEL.build_vocab(train_dataset, dev_dataset,test_dataset)

	train_iter, dev_iter,test_iter = 
	torchtext.data.BucketIterator.splits(
        (train_dataset, dev_dataset,test_dataset),   #需要生成迭代器的数据集
        batch_sizes=(args.batch_size, args.batch_size,args.batch_size),                  # 每个迭代器分别以多少样本为一个batch,验证集和测试集数据不需要训练，全部放在一个batch里面就行了
        sort_key=lambda x: len(x.content)            #按什么顺序来排列batch，这里是以句子的长度，就是上面说的把句子长度相近的放在同一个batch里面
        )

	return TEXT,LABEL,train_iter,dev_iter,test_iter

