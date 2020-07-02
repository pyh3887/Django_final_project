from django.shortcuts import render
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font',family='malgun gothic')   #한글 깨짐 방지.
plt.rcParams['axes.unicode_minus'] = False   # -부호 깨짐 방지

def mainFunc(request):  
    studentData = pd.read_csv('https://raw.githubusercontent.com/pyh3887/Django_final_project/master/student.csv',encoding='cp949')
#     print(studentData.head(3))
    studentData.성적.value_counts()
    sns.countplot(x='부모의학교만족도',data = studentData, hue='성적',palette='bright')
#     fig = plt.gcf()
#     fig.savefig('C:/work/py_sou/chartdb/mychart/static/image/chart.png', dpi=fig.dpi)
    
    return render(request,'chart.html')
