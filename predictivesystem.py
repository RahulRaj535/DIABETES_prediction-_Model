# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle
loaded_model= pickle.load(open('C:/Users/ranja/Downloads/trained_model_diabetic.sav','rb'))
input_data=(0,137,40,35,168,43.1,2.288,33)
#converting input to numpy array
numpy_array=np.asarray(input_data)
#reshaping for predicting for ome data
input_data_reshape=numpy_array.reshape(1,-1)
#standarized datta
#std_data=scaler.transform(input_data_reshape)
#print(std_data)
prediction=loaded_model.predict(input_data_reshape)
print(prediction)

if (prediction[0]==0):
    print('the person is non_diabetic')
else:
     print('the person is diabetic')



