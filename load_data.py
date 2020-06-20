import torch as t
import torchvision as tv
from torch.utils.data import Dataset
from torchvision import transforms
import os
import json
import numpy as np
import  pandas as pd
from torch.utils.data import ConcatDataset
# import matplotlib.pyplot as plt
# class Myconcatdataset(Dataset):
# 	def __init__(self):
class Mytraindataset(Dataset):
	def __init__(self,id):	# 读取文件名
		self.id = id
		self.config = json.load(open('package.json'))
		self.root_dir = self.config["training_path"]	# 训练集的路径
		self.name = np.array(os.listdir(self.root_dir))[id]  # 将训练集路径下的文件名全部读取并存为array
		self.label_path = self.config["label_path"]  #标签路径
		self.read_label()
		self.init_transform()

	def __len__(self):		 # 返回数据集的尺寸
		return len(self.name)

	def __getitem__(self, idx):  # 通过idx获取索引的数据
		data = np.load(os.path.join(self.root_dir,self.name[idx]))  # 路径拼接并读取

		voxel = self.transform(data['voxel'].astype(np.float32)/255)
		voxel =  voxel[34:66,34:66,34:66]
		# voxel = voxel.transpose(0,1).flip(0)
		voxel = voxel.unsqueeze(0)

		seg = self.transform(data['seg'].astype(np.float32))
		seg = seg[34:66,34:66,34:66]
		# seg = seg.transpose(0,1).flip(0)
		seg = seg.unsqueeze(0)


		label = self.label[idx]
		data = np.concatenate([voxel,seg])
		# data = voxel
		return data,label  # 其实是对应idx的返回值

	def init_transform(self):
		self.transform = transforms.Compose([
			# transforms.ToPILImage(),
			# transforms.RandomCrop(32, padding=4),
			# transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			# transforms.Normalize(mean=[.5], std=[.5])


		]) # 转换为totensor

	def	read_label(self):
		dataframe = pd.read_csv(self.label_path)
		dataframe = np.array(dataframe)
		self.label = dataframe[self.id,1]
class Mytraindataset_roate(Dataset):
	def __init__(self,id):	# 读取文件名
		self.id = id
		self.config = json.load(open('package.json'))
		self.root_dir = self.config["training_path"]	# 训练集的路径
		self.name = np.array(os.listdir(self.root_dir))[id]  # 将训练集路径下的文件名全部读取并存为array
		self.label_path = self.config["label_path"]  #标签路径
		self.read_label()
		self.init_transform()

	def __len__(self):		 # 返回数据集的尺寸
		return len(self.name)

	def __getitem__(self, idx):  # 通过idx获取索引的数据
		data = np.load(os.path.join(self.root_dir,self.name[idx]))  # 路径拼接并读取

		voxel = self.transform(data['voxel'].astype(np.float32)/255)
		voxel =  voxel[34:66,34:66,34:66]
		# voxel = voxel.transpose(0,1).flip(0)
		voxel = voxel.flip(0).flip(1)
		voxel = voxel.unsqueeze(0)

		seg = self.transform(data['seg'].astype(np.float32))
		seg = seg[34:66,34:66,34:66]
		# seg = seg.transpose(0,1).flip(0)
		seg = seg.flip(0).flip(1)
		seg = seg.unsqueeze(0)


		label = self.label[idx]
		data = np.concatenate([voxel,seg])
		# data = voxel
		return data,label  # 其实是对应idx的返回值

	def init_transform(self):
		self.transform = transforms.Compose([
			# transforms.ToPILImage(),
			# transforms.RandomCrop(32, padding=4),
			# transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			# transforms.Normalize(mean=[.5], std=[.5])


		]) # 转换为totensor

	def	read_label(self):
		dataframe = pd.read_csv(self.label_path)
		dataframe = np.array(dataframe)
		self.label = dataframe[self.id,1]

class Mytraindataset_roate_2(Dataset):
	def __init__(self,id):	# 读取文件名
		self.id = id
		self.config = json.load(open('package.json'))
		self.root_dir = self.config["training_path"]	# 训练集的路径
		self.name = np.array(os.listdir(self.root_dir))[id]  # 将训练集路径下的文件名全部读取并存为array
		self.label_path = self.config["label_path"]  #标签路径
		self.read_label()
		self.init_transform()

	def __len__(self):		 # 返回数据集的尺寸
		return len(self.name)

	def __getitem__(self, idx):  # 通过idx获取索引的数据
		data = np.load(os.path.join(self.root_dir,self.name[idx]))  # 路径拼接并读取

		voxel = self.transform(data['voxel'].astype(np.float32)/255)
		voxel =  voxel[34:66,34:66,34:66]
		voxel = voxel.transpose(0,1).flip(0)
		# voxel = voxel.flip(0).flip(1)
		voxel = voxel.unsqueeze(0)

		seg = self.transform(data['seg'].astype(np.float32))
		seg = seg[34:66,34:66,34:66]
		seg = seg.transpose(0,1).flip(0)
		# seg = seg.flip(0).flip(1)
		seg = seg.unsqueeze(0)


		label = self.label[idx]
		data = np.concatenate([voxel,seg])
		# data = voxel
		return data,label  # 其实是对应idx的返回值

	def init_transform(self):
		self.transform = transforms.Compose([
			# transforms.ToPILImage(),
			# transforms.RandomCrop(32, padding=4),
			# transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			# transforms.Normalize(mean=[.5], std=[.5])


		]) # 转换为totensor

	def	read_label(self):
		dataframe = pd.read_csv(self.label_path)
		dataframe = np.array(dataframe)
		self.label = dataframe[self.id,1]

class Mytestdataset(Dataset):
	def __init__(self,id):	# 读取文件名
		self.id = id
		self.config = json.load(open('package.json'))
		self.root_dir = self.config["training_path"]	# 训练集的路径
		self.name = np.array(os.listdir(self.root_dir))[id]  # 将训练集路径下的文件名全部读取并存为array
		self.label_path = self.config["label_path"]  #标签路径
		self.read_label()
		self.init_transform()

	def __len__(self):		 # 返回数据集的尺寸
		return len(self.name)

	def __getitem__(self, idx):  # 通过idx获取索引的数据
		data = np.load(os.path.join(self.root_dir,self.name[idx]))  # 路径拼接并读取

		voxel = self.transform(data['voxel'].astype(np.float32)/255)
		voxel =  voxel[34:66,34:66,34:66]
		# voxel = voxel.transpose(0,1).flip(0)
		voxel = voxel.unsqueeze(0)

		seg = self.transform(data['seg'].astype(np.float32))
		seg = seg[34:66,34:66,34:66]
		# seg = seg.transpose(0,1).flip(0)
		seg = seg.unsqueeze(0)


		label = self.label[idx]
		data = np.concatenate([voxel,seg])
		# data = voxel
		return data,label  # 其实是对应idx的返回值

	def init_transform(self):
		self.transform = transforms.Compose([
			# transforms.ToPILImage(),
			# transforms.RandomCrop(32, padding=4),
			# transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			# transforms.Normalize(mean=[.5], std=[.5])


		]) # 转换为totensor

	def	read_label(self):
		dataframe = pd.read_csv(self.label_path)
		dataframe = np.array(dataframe)
		self.label = dataframe[self.id,1]
class Mlset():
	def __init__(self):
		super().__init__()
		self.config = json.load(open('package.json'))
		self.root_dir = self.config["training_path"]  # 训练集的路径
		self.name = np.array(os.listdir(self.root_dir))

	def train_test_split(self,k=0.8):
		length = len(self.name)
		id = np.array(range(length))
		np.random.shuffle(id)
		self.train_id = id[:(int)(length*k)]
		self.test_id = id[(int)(length*k):]

		self.train_set_1 = Mytraindataset(self.train_id)
		self.train_set_roate = Mytraindataset_roate(self.train_id)
		self.train_set_roate_2 = Mytraindataset_roate_2(self.train_id)
		self.test_set = Mytestdataset(self.test_id)

		return self.train_set_1,self.train_set_roate,self.train_set_roate_2,self.test_set

class Restset(Dataset):
	def __init__(self):
		super().__init__()
		self.config = json.load(open('package.json'))
		self.test_path = self.config["test_path"]
		self.test_name = (os.listdir(self.test_path))
		self.init_transform()

	def __len__(self):
		return len(self.test_name)

	def __getitem__(self, idx):
		data = np.load(os.path.join(self.test_path,self.test_name[idx]))

		voxel = self.transform(data['voxel'].astype(np.float32)/255)
		voxel = voxel[34:66,34:66,34:66]
		voxel = voxel.unsqueeze(0)

		seg = self.transform(data['seg'].astype(np.float32))
		seg = seg[34:66,34:66,34:66]
		seg = seg.unsqueeze(0)

		data = np.concatenate([voxel,seg])
		# data = voxel
		name = self.test_name[idx]
		return data,name

	def init_transform(self):
		self.transform = transforms.Compose([
			transforms.ToTensor(),
			# transforms.Normalize(mean=[.5], std=[.5])
		])

	def sort(self):
		d = self.test_name
		sorted_key_list = sorted(d, key=lambda x: (int)(os.path.splitext(x)[0].strip('candidate')))
		self.test_name = np.array(sorted_key_list) # 排序



dataset = Mlset()
train_set_1,train_set_roate,train_set_roate_2,test_set = dataset.train_test_split()
# train_dataloader = t.utils.data.DataLoader(ConcatDataset([train_set_1,train_set_roate,train_set_roate_2]),batch_size=10,shuffle=True,num_workers=2)
train_dataloader = t.utils.data.DataLoader(ConcatDataset([train_set_1]),batch_size=10,shuffle=True,num_workers=2)

test_dataloader = t.utils.data.DataLoader(test_set,batch_size=4,shuffle=False,num_workers=2)
rest = Restset()

if __name__=="__main__":
	print(len(train_set_1))
	print(len(test_set))
	print(train_set_roate.__len__())
	print(rest.__len__())
	print(ConcatDataset([train_set_1,train_set_roate]).__len__())


	a,b = train_set_roate[1]
	print(a.shape,b)
	rest.sort()
	c,d = rest[1]
	print(c.shape,d)
	# rest.sort()
	# print(rest.test_name)

	# plt.imshow(c[1][15])
	# plt.show()
	# print(c[1].shape)






