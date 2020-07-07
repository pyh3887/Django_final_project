from django.shortcuts import render
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.offline import plot    
import plotly.graph_objs as go
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing._label import LabelEncoder

plt.rc('font',family='malgun gothic')   #한글 깨짐 방지.
plt.rcParams['axes.unicode_minus'] = False   # -부호 깨짐 방지

def mainFunc(request):  
    studentData = pd.read_csv('https://raw.githubusercontent.com/pyh3887/Django_final_project/master/student.csv',encoding='cp949')
#     print(studentData.head(3))
#     studentData.성적.value_counts()

#     plot_div = sns.countplot(x='부모의학교만족도',data = studentData, hue='성적',palette='bright')
#     fig = plt.gcf()  
#     fig.savefig('C:/work/py_sou/chartdb/mychart/static/image/chart.png', dpi=fig.dpi)
#     df = studentData.groupby(['성적','부모의학교만족도'])
    print(studentData.groupby(['성적'])['부모의학교만족도'].size())
    
    fig = go.Figure(data=[
    go.Bar(name='Good', x=['H','L','M'],y=studentData[studentData['부모의학교만족도']=='Good'].groupby(['성적','부모의학교만족도']).size(),
            marker_color='#9bb1d6'),
    go.Bar(name='Bad', x=['H','L','M'],y=studentData[studentData['부모의학교만족도']=='Bad'].groupby(['성적','부모의학교만족도']).size(),
           marker_color='#a39bd6'),
     
    ])
    # Change the bar mode
    fig = fig.update_layout(barmode ='group',width=500,
    height=500,
    title='부모의학교만족도에 따른 성적 그래프',
    xaxis_title='성적',
    yaxis_title='합계',plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',font=dict(family='Courier New, monospace')
    )  #데이터를 그룹화하여 표에 적용

#     fig = px.bar(studentData, x='성적', y='부모의학교만족도', barmode='group',height=400)
    
    plot_div = plot(fig,output_type='div')
    
    #전공별 비율
    fig2 = go.Figure(data=[go.Pie(labels=['Arabic','Biology','Chemistry','English','French','Geology','History','IT','Math','Quran','Science','Spanish'], values=studentData.groupby(['전공']).size(), textinfo='label+percent',
                             insidetextorientation='radial'
                            )])   
    fig2 = fig2.update_layout(width=500,height=600)                          
    pie_div = plot(fig2,output_type='div') 
#     print('룰루랄ㄹ라\n',studentData.groupby(['전공']).size())
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

    fig3.update_layout(title="전공에 따른 성적 그래프",
                  xaxis_title="학생수",
                  yaxis_title="전공",width=700,height=500)
    
    last_div = plot(fig3,output_type='div')
    
    fig4 = go.Figure(data=[go.Bar(name='7일이하',
            x=['H','L','M'], y=studentData[studentData['결석일수']=='Under-7'].groupby(['성적']).size(),
            text=studentData[studentData['결석일수']=='Under-7'].groupby(['성적']).size(),
            textposition='auto',
            marker_color='rgb(204,153,153)'
        ),
        go.Bar(name='7일이상',
            x=['H','L','M'], y=studentData[studentData['결석일수']=='Above-7'].groupby(['성적']).size(),
            text=studentData[studentData['결석일수']=='Above-7'].groupby(['성적']).size(),
            textposition='auto',
            marker_color='rgb(255,204,204)'
        )
        ])
    print(studentData[studentData['결석일수']=='Above-7'].groupby(['성적']).size())
    fig4.update_layout(title="결석일수에 따른 성적 분포",
                  xaxis_title="성적",
                  yaxis_title="학생수",width=500,height=500,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(family='Courier New, monospace', color='#000'))
    
    ab_plot = plot(fig4,output_type='div')
                
    fig5 = px.scatter_matrix(studentData,
    dimensions=["발표수", "과정반복수", "새공지사항확인수", "토론참여수"],
    color="성적",width=1200, height=1000)
    
    plot5_div = plot(fig5,output_type='div')
    
    fig6 = px.strip(studentData, x="발표수", y="발표수", orientation="v", color="성적")

    plot6_div = plot(fig6,output_type='div')






















#찬규씨
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
#     print('발표 수 : \n', data.발표수.value_counts())
    
    # counts 값 percentage 로 변경
#     print(data['국적'].unique())
    # for i in data['국적'].unique():
    
#     for i in len(data['국적'].value_counts()):
#         i = i / 480 * 100
        

    # 이미지 저장 작업
    # plt.gcf()

    # 1행 1열에 국적별 성적 출력
    df = pd.DataFrame({"국적":data['국적'],"성적":data['성적']})
    df7 = pd.crosstab(df.성적, df.국적, margins=True)
#     print(df7.columns)

    for i in df7.columns:
        df7[i] = df7[i].values / df7.loc['All', i] * 100
    # result = df7['Egypt'].values / df7.loc['All', 'Egypt'] * 100
    # print(result)
    df7 = df7.drop(['All'])
    fig8 = go.Figure(data=[
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
    fig8.update_layout(barmode ='stack')
    plot10_div = plot(fig8,output_type='div') 
    
    fig9 = px.violin(data, y="과정반복수", x="결석일수", color="성별", box=True, points="all", hover_data=df.columns)
    fig9.update_layout(width=1000)
    plot11_div = plot(fig9,output_type='div') 


    fig = px.scatter_3d(data, x='발표수', y='토론참여수', z='새공지사항확인수',
              color='성적', size_max=1,
              symbol='성적', opacity=0.7)
    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),width=500,height=500)
    plot12_div = plot(fig,output_type='div')
    # =================================




    #경석씨
    df = data
    print(data.columns)
    # LabelEncoder
    le = LabelEncoder()
    # apply "le.fit_transform"
    df = df.apply(le.fit_transform)
    # 성적 순으로 숫자를 재배치
    df.loc[df['성적'] == 0, '성적'] = 3
    df.loc[df['성적'] == 2, '성적'] = 2
    df.loc[df['성적'] == 1, '성적'] = 1
    print(df['성적'].head())
#     print(df)
    print(df['발표수'].head(20))
    print(np.corrcoef(df['발표수'], df['국적']))
    print(df.corr())
#     피어슨의 상관계수는 일반적으로,
#     값이 -1.0 ~ -0.7 이면, 강한 음적 상관관계
#     값이 -0.7 ~ -0.3 이면, 뚜렷한 음적 상관관계
#     값이 -0.3 ~ -0.1 이면, 약한 음적 상관관계
#     값이 -0.1 ~ +0.1 이면, 없다고 할 수 있는 상관관계
#     값이 +0.1 ~ +0.3 이면, 약한 양적 상관관계
#     값이 +0.3 ~ +0.7 이면, 뚜렷한 양적 상관관계
#     값이 +0.7 ~ +1.0 이면, 강한 양적 상관관계로 해석됩니다.

    fig20 = px.imshow(df.corr(),
                    x=['성별', '국적', '출생지', '교육단계', '학년', '학급', '전공', '학기', '담당부모', '발표수', '과정반복수', '새공지사항확인수', '토론참여수', '부모의학교만족도', '결석일수', '성적'],
                    y=['성별', '국적', '출생지', '교육단계', '학년', '학급', '전공', '학기', '담당부모', '발표수', '과정반복수', '새공지사항확인수', '토론참여수', '부모의학교만족도', '결석일수', '성적'],
                    width=1000, height=900, color_continuous_scale='RdBu_r'
               )
    plot20_div = plot(fig20,output_type='div')


#     print(studentData[studentData['성적']=='M'].groupby(['전공']).size())
    return render(request,'index.html', context={'plot_div': plot_div,'pie_div': pie_div,'last_div': last_div,'ab_plot': ab_plot,'plot5_div': plot5_div,'plot6_div': plot6_div,'plot10_div': plot10_div,'plot11_div': plot11_div,'plot12_div': plot12_div,'plot20_div': plot20_div})
   

