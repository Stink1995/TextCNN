# -*- coding:utf-8 -*-

class Config(object):
	def __init__(self,steps_show=10,
		steps_eval=100,early_stopping=1000,
		data_path = './data/cnews',
		data_format='tsv',
		train_name = 'train.tsv',
		dev_name = 'dev.tsv',
		test_name = 'test.tsv',
		pretrained_name='sgns.sogou.word',
		pretrained_path='./data/wordVectors',
		stopwords_path = "./data/stopwords/哈工大停用词表.txt",
		save_dir = './model'):
		super(Config, self).__init__()

		self.steps_show = steps_show          #每训练多少步打印训练loss和accuracy信息
		self.steps_eval = steps_eval          #每训练多少步进行一次验证集的验证
		self.early_stopping = early_stopping  #验证集获取到best accuracy之后停止训练
		self.data_path = data_path			  #数据集存放路径
		self.data_format = data_format        #文件格式
		self.train_name = train_name          #训练集文件名
		self.dev_name = dev_name              #验证集文件名
		self.test_name = test_name            #测试集文件名
		self.pretrained_name = pretrained_name  #预训练词向量文件名
		self.pretrained_path = pretrained_path  #预训练词向量路径
		self.stopwords_path = stopwords_path  #停用词路径
		self.save_dir = save_dir              #文件保存路径