import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import EditedNearestNeighbours
import pandas as pd 
import seaborn as sn
import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow import keras
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Input
from keras.models import Model, load_model
from tqdm import tqdm
#import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest,SelectFpr,SelectPercentile,SelectFdr,VarianceThreshold, chi2



#metodos=['RNAC','PCA','ENN','RNAC+ENN','PCA+ENN']
metodos=['']

for metodo in range (len(metodos)):
	archivos=["PaviaU_Training_Bloques_Smote","PaviaU_Training_Complete","PaviaU_Training_Complete_Smote"]
	save=str('/home/carlos/Escritorio/Estancia/PaviaU/Resultados/Prueba'+str(metodos[metodo])+"/")

	for archivo in range(len(archivos)):
		for repeticion in range(5):
			data_training=np.loadtxt('/home/carlos/Escritorio/Estancia/PaviaU/Archivos/'+str(archivos[archivo])+"_"+str(repeticion))
			print(data_training.shape)
			data_test=np.loadtxt('/home/carlos/Escritorio/Estancia/PaviaU/Archivos/PaviaU_Test'+"_"+str(repeticion))
			print(data_test.shape)

			x1=data_training[:,0:data_training.shape[1]-1]
			y1=data_training[:,data_training.shape[1]-1]

			x2=data_test[:,0:data_test.shape[1]-1]
			y2=data_test[:,data_test.shape[1]-1]

			min_max_scaler = preprocessing.MinMaxScaler()
			x1 = min_max_scaler.fit_transform(x1)
			x2 = min_max_scaler.transform(x2)

			dim=x1.shape[1]
			dim1=int(dim/3)
			############################
			#preposesing area
			
			
			if metodos[metodo]=='RNAC':
				#Autoencoder
				capa_entrada= Input(shape=(dim,))
				encoder= Dense(dim1, activation='sigmoid')(capa_entrada)
				decoder= Dense(dim, activation= 'sigmoid')(encoder)
				autoencoder = Model(inputs=capa_entrada,outputs=decoder)
				encoder1= Model(inputs=capa_entrada, outputs=encoder)
				sgd= SGD(lr=0.01)
				autoencoder.compile(optimizer='sgd',loss= 'mse')
				autoencoder.fit(x1,x1, epochs= 250, batch_size=1000, verbose=1)
				x1= encoder1.predict(x1)
				x2= encoder1.predict(x2)

			#############################

			elif metodos[metodo]=='PCA':
				#PCA
				print(x1.shape)
				pca=PCA(n_components=dim1)
				pca.fit(x1)
				x1=pca.transform(x1)
				x2=pca.transform(x2)
				print(x1.shape)

			############################			
			elif metodos[metodo]=='ENN':
				enn = EditedNearestNeighbours(sampling_strategy='all',n_jobs=7)
				x1, y1 = enn.fit_resample(x1, y1)

			############################
			elif metodos[metodo]=='RNAC+ENN':
				capa_entrada= Input(shape=(dim,))
				encoder= Dense(dim1, activation='sigmoid')(capa_entrada)
				decoder= Dense(dim, activation= 'sigmoid')(encoder)
				autoencoder = Model(inputs=capa_entrada,outputs=decoder)
				encoder1= Model(inputs=capa_entrada, outputs=encoder)
				sgd= SGD(lr=0.01)
				autoencoder.compile(optimizer='sgd',loss= 'mse')
				autoencoder.fit(x1,x1, epochs= 250, batch_size=1000, verbose=1)
				x1= encoder1.predict(x1)
				x2= encoder1.predict(x2)

				enn = EditedNearestNeighbours(sampling_strategy='all',n_jobs=7)
				x1, y1 = enn.fit_resample(x1, y1)

			############################
			elif metodos[metodo]=='PCA+ENN':
				print(x1.shape)
				pca=PCA(n_components=dim1)
				pca.fit(x1)
				x1=pca.transform(x1)
				x2=pca.transform(x2)

				enn = EditedNearestNeighbours(sampling_strategy='all',n_jobs=7)
				x1, y1 = enn.fit_resample(x1, y1)

			

			#NeuralNetwork
			#Indian 500, Salinas 500, PaviaU 250, Pavia 250, Botswana 250, KSC 250
			epochs=250

			#Indian 500, Salinas 500, PaviaU 1000, Pavia 1000, Botswana 1000, KSC 1000
			batch_size=1000

			#indian 17, Salinas 17, PaviaU 10, Pavia 10, Botswana 15, KSC 14
			numero_clases=10


			tf.keras.backend.clear_session() 
			capa_entrada= Input(shape=(x1.shape[1],))
			capa1= Dense(50, activation='relu')(capa_entrada)
			capa2= Dense(40, activation='relu')(capa1)
			capa3= Dense(30, activation='relu')(capa2)
			capa4= Dense(20, activation='relu')(capa3)
			capa_salida= Dense(numero_clases, activation='softmax')(capa4)
			salida = Model(inputs=capa_entrada,outputs=capa_salida)

			adam=optimizers.Adam(lr=.001)
			salida.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
			snn=salida.fit(x1,y1,validation_data=(x2,y2),batch_size=batch_size,epochs=epochs,shuffle=True, verbose=True)


			plt.plot(snn.history['loss'],label='loss',marker="x", markersize="5", markeredgewidth="2")
			plt.plot(snn.history['val_loss'],label='val_loss',marker="x", markersize="5", markeredgewidth="2")
			sn.set(font_scale=.8)
			plt.title("Loss")
			plt.legend(loc='center right',fontsize='10')
			plt.savefig(str(save+"Loss_"+archivos[archivo]+"_"+str(repeticion)+".png"), bbox_inches='tight')
			plt.close("all")
			#plt.show()

			plt.plot(snn.history['accuracy'],label='accuracy',marker="x", markersize="5", markeredgewidth="2")
			plt.plot(snn.history['val_accuracy'],label='val_accuracy',marker="x", markersize="5", markeredgewidth="2")
			sn.set(font_scale=.8)
			plt.title("Accuracy")
			plt.legend(loc='center right',fontsize='10')
			plt.savefig(str(save+"Acc_"+archivos[archivo]+"_"+str(repeticion)+".png"), bbox_inches='tight')
			plt.close("all")
			#plt.show()
			#Evaluation Test
			evaluation = salida.evaluate(x2,y2,batch_size=batch_size,verbose=False)
			snn_pred = salida.predict(x2, batch_size=batch_size) 
			snn_predicted = np.argmax(snn_pred, axis=1)
			snn_cm = confusion_matrix(y2, snn_predicted) 
			snn_cmN= np.zeros((len(snn_cm),len(snn_cm)))

			for i in range(len(snn_cm)):
				total=0
				for k in range(len(snn_cm)):
					total=total+snn_cm[i][k]
					total=total.astype(float)
				for j in range(len(snn_cm)):
					snn_cmN[i][j]=np.round(snn_cm[i][j]/total,4)

			snn_df_cm = pd.DataFrame(snn_cmN, range(numero_clases), range(numero_clases))
			plt.figure(figsize = (16,10)) 
			sn.set(font_scale=1.4) #for label size 
			sn.heatmap(snn_df_cm, annot=True,annot_kws={"size": 12},cmap="Greys", robust=True,center=True)
			plt.savefig(str(save+"Matrix_"+archivos[archivo]+"_"+str(repeticion)+".png"), bbox_inches='tight')
			plt.close("all")

			ArchivoReport = open(str(save+archivos[archivo]+"_"+str(repeticion)),'w')
			snn_report1 = classification_report(y2, snn_predicted,digits=4)
			ArchivoReport.write(snn_report1)
			ArchivoReport.write(os.linesep)
			ArchivoReport.close()
			print(snn_report1)

