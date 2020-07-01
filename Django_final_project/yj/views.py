from django.shortcuts import render
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create your views here.
def mainFunc(request):  
    gender = pd.read_csv('C:\work\git\Django_final_project\Django_final_project\yj\smart.csv',encoding='euc-kr') 
#     print(gender.head(3))
    male = gender[(gender['gen'] == 1)].shape[0]
    female = gender[(gender['gen'] == 2)].shape[0]
    print(male)
    print(female)
#     print(gender.head(3))
    df = pd.DataFrame({ 'male': [male], 'female': [female] })
    
    return render(request,'aa.html')
