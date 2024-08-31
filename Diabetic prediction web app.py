# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 23:43:57 2024

@author: ranja
"""

import numpy as np
import pickle
import streamlit as st

loaded_model= pickle.load(open('C:/Users/ranja/Downloads/trained_model_diabetic.sav','rb'))

def diabeties_prediction(input_data):
   # input_data=(0,137,40,35,168,43.1,2.288,33)
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
        return'the person is non_diabetic'
    else:
         return'the person is diabetic'
         
         
def main():
    st.title('Diabetes prediction ')
    #'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
      # 'BMI', 'DiabetesPedigreeFunction', 'Age',
    Pregnancies=st.text.input('Number of pregnancies')
    Glucose=st.text.input('Glucose level')
    BloodPressure=st.text.input('Blood Pressure level')
    SkinThickness=st.text.input(' Skin Thickness level')
    Insulin=st.text.input(' Insulin level')
    BMI=st.text.input(' BMI level')
    DiabetesPedigreeFunction=st.text.input(' Diabetes Pedigree Function level')
    Age=st.text.input(' Age of person')
    
    
    
    # code for prediction
    diagonses=''
    # creating a button for prediction 
    if st.button('show Diabetes Result'):
        diagonses= diabeties_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
   
    st.success (diagonses)     
  
  
if_name_=='_main_':
    main()
    









         
