import scipy
from scipy.io import loadmat 
from collections import Counter
import numpy as np
import os

import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap



N = 14
vals = np.ones((N, 4))
colores=np.array([[0.8,0.8,0.8],[1,0,0],[0,1,0],[0,0,1],[0,0,0],[1,1,0],[0,1,1],[1,0,1],[0.7,0.7,0.5],[.5,0,0],[0,.5,0],[0,0,.5],[0.5,0.5,0],[0,0.5,0.5],[.5,0,0.5],[1,.5,0],[1,0.5,0.5],[1,1,1],[0.5,1,0.5]])
for i in range (N):
  vals[i,0]=colores[i][0]
  vals[i,1]=colores[i][1]
  vals[i,2]=colores[i][2]
newcmp = ListedColormap(vals)



#load Class and data
clas=loadmat("/home/carlos/Escritorio/Estancia/KSC/KSC_gt.mat")
print(clas)
print(clas['KSC_gt'].shape)
data= loadmat("/home/carlos/Escritorio/Estancia/KSC/KSC.mat")
print(data['KSC'].shape)


save_img="/home/carlos/Escritorio/Estancia/KSC/Imagenes/"
save_data="/home/carlos/Escritorio/Estancia/KSC/Archivos/DataKSC"
save_clas="/home/carlos/Escritorio/Estancia/KSC/Archivos/ClasKSC"

#set section parameters
#Indian 35, Salinas 100, PaviaU 100, Botswana 125, KSC=100
section_len=102
x_start=0
y_start=0
x_end=x_start+section_len
y_end=y_start+section_len


#show the imagen section distribution
plt.imshow(clas['KSC_gt'],newcmp)
plt.grid(section_len)
for x_section in range (int(clas['KSC_gt'].shape[0]/section_len)):
	y_start=0
	for y_section in range(int(clas['KSC_gt'].shape[1]/section_len)):
		plt.text(y_start+(20), x_start+(20), str("("+str(x_section)+","+str(y_section))+")", backgroundcolor='white',fontsize=6)
		y_start=y_start+section_len
	x_start=x_start+section_len
plt.savefig(str(save_img+"Figure_All.png"), bbox_inches='tight')
plt.close("all")

x_start=0
y_start=0
x_end=x_start+section_len
y_end=y_start+section_len

def clas_matrix_transformation(clas_section):
	matrix=np.zeros(clas_section.shape[0]*clas_section.shape[1])
	contador=0
	for col in range (clas_section.shape[0]):
		for row in range (clas_section.shape[1]):
			matrix[contador]=clas_section[col][row]
			contador=contador+1
	return matrix

def data_matrix_transformation(data_section):
	matrix=np.zeros((data_section.shape[0]*data_section.shape[1],data_section.shape[2]))
	contador=0
	for col in range (data_section.shape[0]):
		for row in range (data_section.shape[1]):
			for band in range (data_section.shape[2]):
				matrix[contador][band]=data_section[col][row][band]
			contador=contador+1
	return matrix

#cut the imagen on diferent section
for x_section in range(int(clas['KSC_gt'].shape[0]/section_len)):
	x_end=x_start+section_len
	y_start=0
	for y_section in range(int(clas['KSC_gt'].shape[1]/section_len)):
		y_end=y_start+section_len
		clas_section=clas['KSC_gt'][x_start:x_end,y_start:y_end]
		data_section=data['KSC'][x_start:x_end,y_start:y_end]
		print(data_section.shape)

		clas_matrix=clas_matrix_transformation(clas_section)
		print(Counter(clas_matrix))
		print(clas_matrix)

		data_matrix=data_matrix_transformation(data_section)
		print(data_matrix.shape)

		data_sampling=data_matrix
		clas_samplimg=clas_matrix

		ArchivoDatosPart = open(str(save_data+"Part"+str(x_section)+str(y_section)),'w')
		ArchivoClasesPart = open(str(save_clas+"Part"+str(x_section)+str(y_section)),'w')


		for row in range(data_matrix.shape[0]):
			for column in range (data_matrix.shape[1]):
				ArchivoDatosPart.write(str(np.round(data_matrix[row][column],0)))
				ArchivoDatosPart.write(" ")
			ArchivoClasesPart.write(str(clas_matrix[row]))
			ArchivoClasesPart.write(os.linesep)
			ArchivoDatosPart.write(os.linesep)
		ArchivoDatosPart.close()
		ArchivoClasesPart.close()
		print(clas_section.shape)

		clas_section_copy=clas_section
		for j in range (N):
			clas_section_copy[0][j]=j

		plt.imshow(clas_section_copy, newcmp)
		plt.text(5,5, str("("+str(x_section)+","+str(y_section))+")", backgroundcolor='white',fontsize=10)
		
		for i in range (N):
			plt.text(section_len+10,i*5+10, str("Class "+str(i)+": "+str(Counter(clas_matrix)[i])))
			plt.text(section_len+5, i*5+10, str("  "), backgroundcolor=colores[i],fontsize=6)
		plt.savefig(str(save_img+"Figure"+str(x_section)+str(y_section)+".png"), bbox_inches='tight')
		plt.close("all")

		y_start=y_start+section_len
	x_start=x_start+section_len

