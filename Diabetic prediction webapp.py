# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 00:10:22 2024

@author: ranja
"""

import numpy as np
import pickle
import streamlit as st


loaded_model = pickle.load(open("C:/ARAHUL COLLEGE/rahul colleges/python projects\Diabeties/trained_model.sav", 'rb'))

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
        return'the person is not a diabetic patient'
    else:
         return'the person is diabetic patient '
         
         
def main():
    st.title('Diabetes prediction ')

    Pregnancies=st.text_input('Number of pregnancies')
    Glucose=st.text_input('Glucose level')
    BloodPressure=st.text_input('Blood Pressure level')
    SkinThickness=st.text_input(' Skin Thickness level')
    Insulin=st.text_input(' Insulin level')
    BMI=st.text_input(' BMI level')
    DiabetesPedigreeFunction=st.text_input(' Diabetes Pedigree Function level')
    Age=st.text_input(' Age of person')
    
    
    
    
    # code for prediction
    diagonses=''
    # creating a button for prediction 
    if st.button('show Diabetes Result'):
        diagonses= diabeties_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
   
    st.success(diagonses)     
  
  
if __name__=='__main__':
    main()
    