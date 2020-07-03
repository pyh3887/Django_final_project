from django.shortcuts import render
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.offline import plot    
import plotly.graph_objs as go
import seaborn as sns
import plotly.express as px

plt.rc('font',family='malgun gothic')   #한글 깨짐 방지.
plt.rcParams['axes.unicode_minus'] = False   # -부호 깨짐 방지

def MainFunc(request):  
    studentData = pd.read_csv('https://raw.githubusercontent.com/pyh3887/Django_final_project/master/student.csv',encoding='cp949')
#     print(studentData.head(3))
#     studentData.성적.value_counts()

#     plot_div = sns.countplot(x='부모의학교만족도',data = studentData, hue='성적',palette='bright')
#     fig = plt.gcf()  
#     fig.savefig('C:/work/py_sou/chartdb/mychart/static/image/chart.png', dpi=fig.dpi)
#     df = studentData.groupby(['성적','부모의학교만족도'])
    print(studentData.groupby(['성적'])['부모의학교만족도'].size())
    
    fig = go.Figure(data=[
    go.Bar(name='Good', x=['H','L','M'],y=studentData[studentData['부모의학교만족도']=='Good'].groupby(['성적','부모의학교만족도']).size()),
    go.Bar(name='Bad', x=['H','L','M'],y=studentData[studentData['부모의학교만족도']=='Bad'].groupby(['성적','부모의학교만족도']).size()),
    ])
    # Change the bar mode
    fig = fig.update_layout(barmode ='group')  #데이터를 그룹화하여 표에 적용

#     fig = px.bar(studentData, x='성적', y='부모의학교만족도', barmode='group',height=400)
    
    plot_div = plot(fig,output_type='div')
    
    #전공별 비율
    fig2 = go.Figure(data=[go.Pie(labels=['IT','Math','Arabic','Science','English','Quran','Spanish','French','History','Biology','Chemistry','Geology'], values=studentData.groupby(['전공']).size(), textinfo='label+percent',
                             insidetextorientation='radial'
                            )])   
                             
    pie_div = plot(fig2,output_type='div')
    
    #전공별 성적 비교----의미가 있는지 없는지 확인받기
    majors = ['Arabic','Biology','Chemistry','English','French','Geology','History','IT','Math','Quran','Science','Spanish']

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
    x=studentData[studentData['성적']=='H'].groupby(['전공']).size(),
    y=majors,
    marker=dict(color="crimson", size=12),
    mode="markers",
    name="High",
    ))

    fig3.add_trace(go.Scatter(
        x=studentData[studentData['성적']=='M'].groupby(['전공']).size(),
        y=majors,
        marker=dict(color="gold", size=12),
        mode="markers",
        name="Middle",
        ))
    
    fig3.add_trace(go.Scatter(
        x=studentData[studentData['성적']=='L'].groupby(['전공']).size(),
        y=majors,
        marker=dict(color="black", size=12),
        mode="markers",
        name="Low",
        ))

    fig3.update_layout(title="Gender Earnings Disparity",
                  xaxis_title="Annual Salary (in thousands)",
                  yaxis_title="School")
    
    last_div = plot(fig3,output_type='div')
#     print(studentData[studentData['성적']=='M'].groupby(['전공']).size())
    return render(request,'chart.html', context={'plot_div': plot_div,'pie_div': pie_div,'last_div': last_div})


