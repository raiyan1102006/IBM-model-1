#!/usr/bin/python3

import numpy as np
from collections import defaultdict
import sys
# import matplotlib
# import matplotlib.pyplot as plt
# plt.rc('figure', figsize = (8,12))

def read_data(file_name,add_null=False):
	all_data = [] 

	with open(file_name,encoding="latin-1") as data_file:
		for a_line in data_file:
			a_line = a_line.strip().split(' ')
			if add_null:
				a_line=['NULL']+a_line # add NULL for English sentences
			all_data.append(a_line)
	return all_data

def calculate_LL(P,all_data_eng,all_data_fra):
	LL = 0.
	for eng_s, fr_s in zip(all_data_eng, all_data_fra):
		for j in range(len(fr_s)):
			temp_ = 0.
			for i in range(len(eng_s)):
				temp_ += P[(fr_s[j],eng_s[i])]/len(eng_s) #eng_s already has NULL in it, so the length is l+1
			LL+=np.log(temp_)
	return LL


if __name__ == '__main__':
	sample_flag = 0

	if len(sys.argv)>1 and sys.argv[1]=='sample':
		train_data_eng = read_data('training_sample.eng',add_null=True)
		train_data_fra = read_data('training_sample.fra')
		test_data_eng = read_data('test_sample.eng',add_null=True)
		test_data_fra = read_data('test_sample.fra')
		sample_flag = 1
	elif len(sys.argv)<=1:
		train_data_eng = read_data('training.eng',add_null=True)
		train_data_fra = read_data('training.fra')
		test_data_eng = read_data('test.eng',add_null=True)
		test_data_fra = read_data('test.fra')
	else:
		print("Incorrect command. Please refer to the README file.")
		exit()


	P = defaultdict(lambda: 10**-6)
	n_iter=15

	LL_train=[]
	LL_test=[]

	for iter_ in range(n_iter):
		print("begin iteration: "+str(iter_))
		t = {}
		t_table = defaultdict(lambda: 0)
		t_table_eng = defaultdict(lambda: 0)

		# E step
		print("begin e-step")
		for sent_id, (eng_s, fr_s) in enumerate(zip(train_data_eng, train_data_fra)):
			if sample_flag:
				print("sent "+str(sent_id)+" "+str(eng_s)+" "+str(fr_s))
			
			total = defaultdict(lambda: 0)
			for j in range(len(fr_s)):
				for i in range(len(eng_s)):
					t[(j,i)] = P[(fr_s[j],eng_s[i])]
					total[j] += P[(fr_s[j],eng_s[i])]
					
					if sample_flag:
						print("prob lookup: " +str(eng_s[i])+" "+str(fr_s[j])+" "+str(t[(j,i)])+" "+str(total[j]))
				
				for i in range(len(eng_s)):
					t[(j,i)] = t[(j,i)] / total[j] 
					t_table[(fr_s[j],eng_s[i])] += t[(j,i)]
					t_table_eng[eng_s[i]] += t[(j,i)]

			
		# Calculate LL
		LL_train_new = calculate_LL(P,train_data_eng,train_data_fra)
		LL_train.append(LL_train_new)
		print("training corpus log likelihood: "+str(LL_train_new))
		
		LL_test_new = calculate_LL(P,test_data_eng,test_data_fra)
		LL_test.append(LL_test_new)
		print("test corpus log likelihood: "+str(LL_test_new))


		# Plotting code
		# plt.figure()
		# plt.subplot(2,1,1);
		# plt.plot(range(1,len(LL_train)),LL_train[1:],'*',color = '#539caf' ,markersize=8,label = 'Training data')
		# plt.xlabel("Iteration")
		# plt.ylabel("Log Likelihood")
		# plt.title("Training data")
		# plt.legend(loc = 'best') 
		# plt.subplot(2,1,2);
		# plt.plot(range(1,len(LL_test)),LL_test[1:],'*',color ='#7663b0',markersize=8,label = 'Test data')
		# plt.xlabel("Iteration")
		# plt.title("Test data")
		# plt.ylabel("Log Likelihood")
		# plt.legend(loc = 'best') 


		# M step
		print("begin m-step")
		for key,value in t_table.items():
			P[key] = t_table[key]/t_table_eng[key[1]]
			if sample_flag:
				print(str(key[1])+" "+str(key[0])+" "+str(t_table[key])+" "+str(P[key]))
		print(" ")

