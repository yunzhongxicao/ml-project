import json
import torch
import numpy as np
import os
import pandas as pd
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F

# from c3d_v1 import C3D
# from  c3d_v2 import C3D
import resnet
# import resnet_v2
# import vggnet_v1
from load_data import *

config = json.load(open('package.json'))
os.environ['CUDA_VISIBLE_DEVICES']='0' # 第3块GPU
DEVICE = torch.device('cuda:0')

testset = Restset()
testset.sort()
# print(testset)
test_loader = data.DataLoader(testset,batch_size=1,shuffle=False)

# optimizer = torch.nn.CrossEntropyLoss()
# test_model = C3D()
test_model = resnet.resnet10(sample_size=8,sample_duration=4)
# test_model = resnet_v2.generate_model(model_depth=10)
# test_model = vggnet_v1.vgg13_bn()
test_model = test_model.to(DEVICE)
test_model.load_state_dict(torch.load('model_new_res.pkl'))
test_model.eval()

with torch.no_grad():
	# Test the test_loader
	Name = []
	Score = []
	Z = []

	for data, name in test_loader:
		data = data.to(DEVICE)
		# data = Variable(data)
		a = test_model(data)
		# print('a',a)
		out1 = F.softmax(a)
		# print("out1",out1)

		out = out1
		# print(out.size())
		# print(out)
		out = out.squeeze()
		# print(out.size())
		# print(out)
		# print(type(name))
		# print(name)
		Name.append(name[0])
		Score.append((out[1]).item())

	for i in Name:
		i = np.char.strip(i, '.npz')
		Z.append(i)

	test_dict = {'name': Z, 'predicted': Score}
	print(type(test_dict))
	test_dict_df = pd.DataFrame(test_dict)


	print(test_dict_df)
	test_dict_df.to_csv('Submission.csv', index=False)

