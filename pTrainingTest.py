import scipy
from scipy.io import loadmat 
from collections import Counter
import numpy as np
import os

import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.model_selection import train_test_split

def saveData (data,clas,archivo):
	save= open(archivo,'a')
	for i in range (data.shape[0]):
		for j in range (data.shape[1]):
			save.write(str(data[i][j]))
			save.write(" ")
		save.write(str(clas[i]))
		save.write(os.linesep)
	save.close()

a=["00","01","02","03","04","05",
"10","11","12","13","14","15",
"20","21","22","23","24","25",
"30","31","32","33","34","35",
"40","41","42","43","44","45"]

for repeticion in range (5):
	for ar in range (len(a)):
		#load Class and data
		print("Part: "+str(a[ar]))
		clas=np.loadtxt("/home/carlos/Escritorio/Estancia/KSC/Archivos/ClasKSCPart"+str(a[ar]))
		print(clas.shape)
		data= np.loadtxt("/home/carlos/Escritorio/Estancia/KSC/Archivos/DataKSCPart"+str(a[ar]))
		print(data.shape)

		x1,x2,y1,y2 = train_test_split(data, clas, test_size=0.3)
		print(Counter(y1))
		print(Counter(y2))
		print(x1.shape, x2.shape)

		dataTest =str("/home/carlos/Escritorio/Estancia/KSC/Archivos/KSC_Test_"+str(repeticion))
		dataTrainingComplete=str("/home/carlos/Escritorio/Estancia/KSC/Archivos/KSC_Training_Complete_"+str(repeticion))
		dataTrainingBloquesSmote=str("/home/carlos/Escritorio/Estancia/KSC/Archivos/KSC_Training_Bloques_Smote_"+str(repeticion))

		saveData(x2,y2,dataTest)
		saveData(x1,y1,dataTrainingComplete)

		try:
			smote=SMOTE(n_jobs=7)
			x_res, y_res = smote.fit_resample(x1, y1)
			print(x_res.shape)
		except:
			x_res=x1
			y_res=y1
		saveData(x_res, y_res, dataTrainingBloquesSmote)

	data= np.loadtxt(dataTrainingComplete)
	x1=data[:,0:data.shape[1]-1]
	y1=data[:,data.shape[1]-1]
	smote=SMOTE(n_jobs=7)
	x_res, y_res = smote.fit_resample(x1, y1)
	dataTrainingCompleteSMOTE=str("/home/carlos/Escritorio/Estancia/KSC/Archivos/KSC_Training_Complete_Smote_"+str(repeticion))
	saveData(x_res,y_res,dataTrainingCompleteSMOTE)