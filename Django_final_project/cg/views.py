from django.shortcuts import render

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly
import seaborn as sns
import os
import plotly.graph_objs as go
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.preprocessing._label import LabelEncoder
from plotly.offline import plot

plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False

# Create your views here.


    
def mainFunc(request):
    plt.clf()
    data = pd.read_csv('https://raw.githubusercontent.com/pyh3887/Django_final_project/master/student.csv', encoding='euc-kr')
    data['성적'] = data['성적'].map({'H':2,'M':1,'L':0})

    # 데이터 상위 5개 행 읽기
    # print(data.head())
    # 데이터의 컬럼별 요약 통계량 확인
    # print(data.describe())
    # 데이터의 행과 열의 개수
    # print(data.shape)
    # 결측값 확인
    # print(data.isnull().sum())
    
    
#     # 국적 인원 분포도 (1)
#     print(data['국적'].value_counts())
#     # 국적 인원 분포도 퍼센트로 막대그래프 출력 (2)
#     print('인원수\n',data.국적.value_counts(normalize=False))
#     plt.subplot(1, 2, 1)
#     data.국적.value_counts(normalize=False).plot(kind='bar')
#     
#     # 국적별 성적 분포도
#     print('인원수\n',data.성적.value_counts(normalize=False))
#     plt.subplot(1, 2, 2)
#     data.성적.value_counts(normalize=False).plot(kind='pie')
    
    # 국적별 발표 수
    print('발표 수 : \n', data.발표수.value_counts())
    
    # counts 값 percentage 로 변경
    print(data['국적'].unique())
    # for i in data['국적'].unique():
    
#     for i in len(data['국적'].value_counts()):
#         i = i / 480 * 100
        

    # 이미지 저장 작업
    # plt.gcf()

    # 1행 1열에 국적별 성적 출력
    df = pd.DataFrame({"국적":data['국적'],"성적":data['성적']})
    df7 = pd.crosstab(df.성적, df.국적, margins=True)
    print(df7.columns)

    for i in df7.columns:
        df7[i] = df7[i].values / df7.loc['All', i] * 100
    # result = df7['Egypt'].values / df7.loc['All', 'Egypt'] * 100
    # print(result)
    df7 = df7.drop(['All'])
    fig = go.Figure(data=[
        go.Bar(name='H', x=['Egypt', 'Iran', 'Iraq', 'Jordan', 'KW', 'Lybia', 'Morocco',
       'Palestine', 'SaudiArabia', 'Syria', 'Tunis', 'USA', 'lebanon',
       'venzuela'],y=df7.iloc[0,:17].values),    
        go.Bar(name='M', x=['Egypt', 'Iran', 'Iraq', 'Jordan', 'KW', 'Lybia', 'Morocco',
       'Palestine', 'SaudiArabia', 'Syria', 'Tunis', 'USA', 'lebanon',
       'venzuela'],y=df7.iloc[1,:17].values),
        go.Bar(name='L', x=['Egypt', 'Iran', 'Iraq', 'Jordan', 'KW', 'Lybia', 'Morocco',
       'Palestine', 'SaudiArabia', 'Syria', 'Tunis', 'USA', 'lebanon',
       'venzuela'],y=df7.iloc[2,:17].values),
        ])
    # Change the bar mode
    fig.update_layout(barmode ='stack')
    plot_div = plot(fig,output_type='div') 
    
    
    import plotly.express as px
    fig = px.violin(data, y="과정반복수", x="결석일수", color="성별", box=True, points="all", hover_data=df.columns)
   
    plot1_div = plot(fig,output_type='div') 
    return render(request,'grade_cuntry_pie.html', context={'plot_div':plot_div,'plot1_div':plot1_div})
