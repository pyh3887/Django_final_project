from django.shortcuts import render
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import  EarlyStopping
import plotly.graph_objs as go 
from plotly.offline import plot 
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn import metrics
from tensorflow.keras import layers
from django.shortcuts import render
import tensorflow as tf
from tensorflow.keras.models import Model
from django.http.response import HttpResponseRedirect, HttpResponse
import json
from sklearn.metrics._classification import accuracy_score
import plotly.express as px
from numpyencoder.numpyencoder import NumpyEncoder

plt.rc('font',family='malgun gothic')
# Create your views here.
def mainFunc(request):
    
    #윤진씨
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
    fig = fig.update_layout(barmode ='group',width=600,
    height=600,
    
    xaxis_title='성적',
    yaxis_title='합계',plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(255,255,255,0)',font=dict(family='Courier New, monospace',color='#fff',size=18)
    )  #데이터를 그룹화하여 표에 적용

#     fig = px.bar(studentData, x='성적', y='부모의학교만족도', barmode='group',height=400)
    
    plot_div = plot(fig,output_type='div')
    
    #전공별 비율
    fig2 = go.Figure(data=[go.Pie(labels=['Arabic','Biology','Chemistry','English','French','Geology','History','IT','Math','Quran','Science','Spanish'], values=studentData.groupby(['전공']).size(), textinfo='label+percent',
                             insidetextorientation='radial'
                            )])   
    fig2 = fig2.update_layout(width=550,height=500,paper_bgcolor='rgba(255,255,255,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(family='Courier New, monospace', color='#fff',size=18),showlegend = False)                          
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

    fig3.update_layout(
                  xaxis_title="학생수",
                  yaxis_title="전공",width=700,height=500,paper_bgcolor='rgba(255,255,255,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(family='Courier New, monospace', color='#fff',size=18))
    
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
    fig4.update_layout(
                  xaxis_title="성적",
                  yaxis_title="학생수",width=600,height=600,paper_bgcolor='rgba(255,255,255,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(family='Courier New, monospace', color='#fff',size=18))
    
    ab_plot = plot(fig4,output_type='div')
                
    fig5 = px.scatter_matrix(studentData,dimensions=["발표수", "과정반복수", "새공지사항확인수", "토론참여수"],color="성적",width=1200, height=1000)
    fig5.update_layout(width=800,height=800,paper_bgcolor='rgba(255,255,255,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(family='Courier New, monospace', color='#fff',size=18))
    
    
    plot5_div = plot(fig5,output_type='div')
    
  


    

    


#찬규씨
    plt.clf()
    data = pd.read_csv('https://raw.githubusercontent.com/pyh3887/Django_final_project/master/student.csv', encoding='euc-kr')
    data['성적'] = data['성적'].map({'H':2,'M':1,'L':0})
   
    
    # 국가별 성적 상.중.하 인원 분포도
    
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
       'venzuela'],y=df7.iloc[0,:17].values,
        marker_color='rgb(152,105,247)'
        ),    
        go.Bar(name='M', x=['Egypt', 'Iran', 'Iraq', 'Jordan', 'KW', 'Lybia', 'Morocco',
       'Palestine', 'SaudiArabia', 'Syria', 'Tunis', 'USA', 'lebanon',
       'venzuela'],y=df7.iloc[1,:17].values,
        marker_color='rgb(247,152,105)'
        ),
        go.Bar(name='L', x=['Egypt', 'Iran', 'Iraq', 'Jordan', 'KW', 'Lybia', 'Morocco',
       'Palestine', 'SaudiArabia', 'Syria', 'Tunis', 'USA', 'lebanon',
       'venzuela'],y=df7.iloc[2,:17].values,
        marker_color='rgb(105,247,152)'
        ),
        ])
    # Change the bar mode
    fig8.update_layout(barmode ='stack',width=900,paper_bgcolor='rgba(255,255,255,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(family='Courier New, monospace', color='#fff',size=18))
    plot10_div = plot(fig8,output_type='div') 
    
    
    #--------- 국가별 성별 비율 그래프 
    
    df1 = pd.DataFrame({"국적":data['국적'],"성별":data['성별']})
    print(df1)
    df8 = pd.crosstab(df1.성별, df1.국적, margins=True)
    print(df8)
    df8 = df8.drop(['All'])
    fig50 = go.Figure(data=[
        go.Bar(name='M', x=['Egypt', 'Iran', 'Iraq', 'Jordan', 'KW', 'Lybia', 'Morocco',
       'Palestine', 'SaudiArabia', 'Syria', 'Tunis', 'USA', 'lebanon',
       'venzuela'],y=df8.iloc[0,:17].values,
        marker_color='rgb(247,105,200)'                    
        ),    
        go.Bar(name='F', x=['Egypt', 'Iran', 'Iraq', 'Jordan', 'KW', 'Lybia', 'Morocco',
       'Palestine', 'SaudiArabia', 'Syria', 'Tunis', 'USA', 'lebanon',
       'venzuela'],y=df8.iloc[1,:17].values,marker_color='rgb(105,200,247)'),
        
        ])
        # Change the bar mode
    fig50.update_layout(barmode ='stack',width=900 ,paper_bgcolor='rgba(255,255,255,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(family='Courier New, monospace', color='#fff',size=18))
    cg_graph = plot(fig50,output_type='div') 
   
   
    
    fig9 = px.violin(data, y="과정반복수", x="결석일수", color="성별", box=True, points="all", hover_data=df.columns)
    fig9.update_layout(width=1300)
    fig9.update_layout(height=300)
    plot11_div = plot(fig9,output_type='div') 
    
    
    
    # 랜덤포레스트 사용
    # 모델 평가 생성
    data = pd.read_csv('https://raw.githubusercontent.com/pyh3887/Django_final_project/master/student.csv', encoding='euc-kr')
    
    df2 = pd.DataFrame({"결석일수":data['결석일수'],"발표수":data['발표수'],"새공지사항확인수":data['새공지사항확인수'],"토론참여수":data['토론참여수']})
    df2['결석일수'] = data['결석일수'].map({'Under-7':0,'Above-7':1})
    
    
    x = df2[['발표수','토론참여수','새공지사항확인수']].values # 2차원
    y = df2[['결석일수']].values # 1차원
    
    # 여기서부터 분류 예측 모델별 생성 후 정확도 분석
    # 모델 평가 생성 - > 정확도 분석
    
    
    # 랜덤포레스트
    new_x = [[0,20,10]]
    
    model = RandomForestRegressor(n_estimators=1000, criterion='mse').fit(x,y)
    modelflatten = model.predict(x)[:10]
    print('예측값 ', np.where(modelflatten.flatten() > 0.5 , 1, 0))
    print('실제값 : ', y[:10].ravel())
    a = model.predict(x)
    xh = np.where(a.flatten() > 0.5,1,0)
    # print(xh.shape)
    # print(y.shape)
    print('RandomForestRegressor : ' , accuracy_score(y,xh))
    print('RandomForestRegressor 새로운 값으로 예측 : ' ,model.predict(new_x))
    print('===========================================================')
    
    
    # LinearRegression
    model1 = LinearRegression().fit(x,y)
    model1flatten = model1.predict(x)[:10]
    print('예측값 ', np.where(model1flatten.flatten() > 0.5 , 1, 0))
    print('실제값 : ', y[:10].ravel())
    a = model1.predict(x)
    xh = np.where(a.flatten() > 0.5,1,0)
    resultLinearRegression = accuracy_score(y,xh)
    print('LinearRegression : ' , resultLinearRegression)
    print('LinearRegression 새로운 값으로 예측 : ' ,model.predict(new_x))
    print('===========================================================')
    
    
    # KNeighborsRegressor
    model2 = KNeighborsRegressor(n_neighbors=3).fit(x,y)
    model2flatten = model2.predict(x)[:10]
    print('예측값 ', np.where(model2flatten.flatten() > 0.5 , 1, 0))
    print('실제값 : ', y[:10].ravel())
    a = model2.predict(x)
    xh = np.where(a.flatten() > 0.5,1,0)
    resultKNeighborsRegressor = accuracy_score(y,xh)
    print('KNeighborsRegressor : ' , accuracy_score(y,xh))
    print('KNeighborsRegressor 새로운 값으로 예측 : ' ,model.predict(new_x))
    print('===========================================================')
    
    
    # XGboost 96%의 확률
    model3 = XGBRegressor(n_estimators=100).fit(x,y)
    model3flatten = model3.predict(x)[:10]
    print('예측값 ', np.where(model3flatten.flatten() > 0.5 , 1, 0))
    print('실제값 : ', y[:10].ravel())
    a = model3.predict(x)
    xh = np.where(a.flatten() > 0.5,1,0)
    resultXGboost = accuracy_score(y,xh)
    print('XGboost : ' , accuracy_score(y,xh))
    print('XGboost 새로운 값으로 예측 : ' ,model.predict(new_x))
    print('===========================================================')
    # = = = = == = = = = = = = 모델별 시각화 
    import plotly.figure_factory as ff
    
    text = [['모델명', '예측값', '실제값', '정확도'], ['RandomForestRegressor',np.where(modelflatten.flatten() > 0.5 , 1, 0), y[:10].ravel(), accuracy_score(y,xh)],
                                            ['LinearRegression',np.where(model1flatten.flatten() > 0.5 , 1, 0), y[:10].ravel(), resultLinearRegression], 
                                            ['KNeighborsRegressor',np.where(model2flatten.flatten() > 0.5 , 1, 0), y[:10].ravel(), resultKNeighborsRegressor], 
                                            ['XGboost',np.where(model3flatten.flatten() > 0.5 , 1, 0), y[:10].ravel(), resultXGboost]]
    
    colorscale = [[0, '#272D31'],[.5, '#ffffff'],[1, '#ffffff']]
    font=['#FCFCFC', 'red', 'black', 'black', 'red']
    
    fig51 = ff.create_table(text, colorscale=colorscale, font_colors=font)
    fig51.layout.width=1300
    cg_chart = plot(fig51,output_type='div') 
    
    
    
    
    
      
   







    
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
    fig20.update_layout(width=800,height=800,paper_bgcolor='rgba(255,255,255,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(family='Courier New, monospace', color='#fff',size=18))
    plot20_div = plot(fig20,output_type='div')
#     print(studentData[studentData['성적']=='M'].groupby(['전공']).size())
    


#--------------------------------------

    #박윤호
    data = pd.read_csv('https://raw.githubusercontent.com/pyh3887/Django_final_project/master/education.csv',encoding='euc-kr')

    print(data)
    label = LabelEncoder()
    Cat_Colums = data.dtypes.pipe(lambda Features: Features[Features=='object']).index
     
    
    for col in Cat_Colums:
        data[col] = label.fit_transform(data[col])
    
    
    
    x = data.drop('성적',axis=1)
    y = data['성적']
    
    print(x)
    print(y)
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=52)
        
    model = XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=100,seed=10)
    fit_model = model.fit(X_train,y_train)
    
    pred = fit_model.predict(X_train)
    
    #yh_acc = accuracy_score(y_train, y_test)
    #print('예측값 : ', pred[:5])
    #print('실제값 : ', np.array(test_y[:5]))
    
    #print('분류 정확도 : ', accuracy_score(test_y, pred))
    feature_important = model.get_booster().get_score(importance_type='weight')
    keys = list(feature_important.keys())
    values = list(feature_important.values())
    print(keys)
    print(values)
    print('중요도:',feature_important)
    #print('특성 중요도 :\n{}'.format(model.feature_importances_))
    print(model)
    
    
   
    
    yh_fig1 = go.Figure(go.Bar(
        x=values,
        y=keys,
        marker=dict(
            color= '#F88137'
            
        ),      
        name='성적과 관련된 중요도 그래프',
        orientation='h',        
        
        
    ))
    
    yh_fig1.update_layout(        
              
        legend=dict(x=0.029, y=1.038, font_size=10),
        margin=dict(l=100, r=20, t=70, b=70),
        paper_bgcolor='rgba(255,255,255,0.2)',
        plot_bgcolor='rgba(255,255,255,0.2)',
        font=dict(color='#000',size=30),
        height = 600
    )
    

   
    yh_fig1.update_layout(yaxis={'categoryorder':'total ascending'})
    
    yh_grap1 = plot(yh_fig1,output_type='div')
    
    
    
    
    
    
    data = pd.read_csv('https://raw.githubusercontent.com/pyh3887/Django_final_project/master/education.csv',encoding='euc-kr')
    yh_fig2 = px.scatter(data, x="발표수", y="토론참여수", color="성적",
                     size='과정반복수', hover_data=['토론참여수']
                    )
    yh_fig2.update_layout(
                          paper_bgcolor='rgba(255,255,255,0.2)',
                          plot_bgcolor='rgba(255,255,255,0.2)',
                          font=dict(color='#000',size=30),
                          height = 550,
                          width = 1400,
                          )
    yh_grap2 = plot(yh_fig2,output_type='div')

    
    data = pd.read_csv('https://raw.githubusercontent.com/pyh3887/Django_final_project/master/student.csv',encoding='euc-kr')
    data['성적'] = data['성적'].map({'H':2,'M':1,'L':0})
    fig = px.scatter_3d(data, x='발표수', y='토론참여수', z='과정반복수',
              color='성적',
              opacity=0.7)
    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),width=600,height=700)
    yh_3D = plot(fig,output_type='div')
    




#--------------------------------------------------------------

    
#----------------------------------------------------------------------------
#경석이형    
    
    
   

    
    
    #-------------------------------------------------------
    # 랜덤포레스트 사용

        
    return render(request,'full.html',{'yh_grap1':yh_grap1,'yh_grap2':yh_grap2,'yh_3D':yh_3D,
    'yj_grap1': plot_div,'yj_pie': pie_div,'yj_grap2': last_div,'yj_grap3': ab_plot,'yj_scatter': plot5_div,'cg_graph1': plot10_div,'cg_graph2': plot11_div,'cg_graph3':cg_graph,'cg_chart':cg_chart,'heatmap': plot20_div})










def ksajax(request):
    data = pd.read_csv('https://raw.githubusercontent.com/pyh3887/Django_final_project/master/education.csv', encoding='euc-kr')
# '성별', '국적', '출생지', '교육단계', '학년', '학급', '전공', '학기', '담당부모', '발표수', '과정반복수', '새공지사항확인수', '토론참여수', '부모의학교만족도', '결석일수', '성적'
    df = data
    #print(data.columns)
    # LabelEncoder
    le = LabelEncoder()
    # apply "le.fit_transform"
    df = df.apply(le.fit_transform)
    # 성적 순으로 숫자를 재배치
    df.loc[df['성적'] == 0, '성적'] = 3
    df.loc[df['성적'] == 2, '성적'] = 2
    df.loc[df['성적'] == 1, '성적'] = 1
    #print(df['성적'].head())
    #print(df)
    #print(df['발표수'].head(20))
    #print(np.corrcoef(df['발표수'], df['국적']))
    #print(df.corr())
   
    
    # 발표 수와 상관관계의 값이 0.3 이상인 것은 담당부모, 과정반복수, 새공지사항확인수, 토론참여수, 부모의학교만족도, 결석일수, 성적이 있다.
    # 그러면 이 칼럼들로 담당부모를 예측해 보겠다.
#     print(df['담당부모'])
    xhas = df.loc[:, ['담당부모', '발표수', '과정반복수', '새공지사항확인수', '토론참여수', '부모의학교만족도', '결석일수', '성적']]
    yhas = df.loc[:, ['발표수']]
    y = df.발표수.tolist()
    
    xt = df.loc[:, ['발표수', '과정반복수', '새공지사항확인수', '토론참여수', '부모의학교만족도', '결석일수', '성적']]
    yt = df.loc[:, ['담당부모']]
    yh = df.담당부모.tolist()

    train_dataset = xhas.sample(frac=0.7, random_state=0)
    test_dataset = xhas.drop(train_dataset.index)
    #print(train_dataset.shape)      # (336, 8)
    #print(test_dataset.shape)       # (144, 8)
    
    train_stat = train_dataset.describe()
    train_stat.pop('발표수')
    train_stat = train_stat.transpose()
    #print(train_stat)       # std열이 표준 편차이다.
    
    # label
    train_labels = train_dataset.pop('발표수')
    #print(train_labels[:2])
    test_labels = test_dataset.pop('발표수')
    #print(test_labels[:2])
    
    def st_func(x): # 표준화 처리 함수    (요소 값 - 평균) / 표준 편차
        return ((x - train_stat['mean']) / train_stat['std'])
    
    #print('st_func(10) :\n', st_func(10))
    st_train_data = st_func(train_dataset)
    st_test_data = st_func(test_dataset)
    
    # 모델 작성 후 예측
    def build_model():
        network = tf.keras.Sequential([
            layers.Dense(64, activation=tf.nn.relu, input_shape=[7]),       # input_shape= 열의 갯수 carname 빠져서 7개
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='linear'),
        ])
        
    #     opti = tf.keras.optimizers.RMSprop(0.001)
        opti = tf.keras.optimizers.Adam(0.01)
        network.compile(optimizer=opti, loss='mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error'])    # mse, mae
        
        return network
    
    model = build_model()
    #print(model.summary())
    
    # fit() 전에 모델을 실행해 볼 수도 있다.
    #print(model.predict(st_train_data[:1]))     # 결과는 신경쓰지 않는다.
    
    # 모델 훈련
    epochs = 500
    
    # 학습 조기 종료
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    
#     history = model.fit(st_train_data,train_labels, epochs=epochs, validation_split=0.2, verbose=1, callbacks=[early_stop])
    history = model.fit(st_train_data,train_labels, epochs=epochs, validation_split=0.2, verbose=1) 
    # 원래 데이터셋이 1000개 이고, fit 함수의 validation_split = 0.2 로 하면, 
    # training dataset 은 800개로 하여, training 시키고,나머지 200개는 test dataset 으로 사용하여, 모델을 평가하게 된다.
    df = pd.DataFrame(history.history)
    #print(df.head())
    #print(df.columns)
    # ['loss', 'mean_squared_error', 'mean_absolute_error',
    #   'val_loss', 'val_mean_squared_error', 'val_mean_absolute_error']    # validation_split을 썻기 때문에 나오는 것이다. 안쓰면 위에 3개만 나온다.
    
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['mean_absolute_error'], name='Train Error',
                             line=dict(color='firebrick', width=4)))
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_mean_absolute_error'], name = 'Val Error',
                             line=dict(color='royalblue', width=4)))
    
    fig.update_layout(title='MAE 오차',
               xaxis_title='Epoch',
               yaxis_title='Mean Abs Error [발표수]',
               width=700, height=700,
               paper_bgcolor='rgba(255,255,255,0.5)',plot_bgcolor='rgba(255,255,255,0.5)',
               font=dict(family='Franklin Gothic', color='#000', size=17))
    plot_div01 = plot(fig, output_type='div')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['mean_squared_error'], name='Train Error',
                             line=dict(color='firebrick', width=4)))
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_mean_squared_error'], name = 'Val Error',
                             line=dict(color='royalblue', width=4)))
    
    fig.update_layout(title='MSE 오차',
               xaxis_title='Epoch',
               yaxis_title='Mean Square Error [발표수]',
               width=700, height=700,
               paper_bgcolor='rgba(255,255,255,0.5)',plot_bgcolor='rgba(255,255,255,0.5)',
               font=dict(family='Franklin Gothic', color='#000', size=17))
    plot_div02 = plot(fig, output_type='div')
    
    # 모델 평가
    loss, mae, mse = model.evaluate(st_test_data, test_labels)
    #print('test dataset으로 모델 평가 mae : {:.3f}'.format(mae))
    #print('test dataset으로 모델 평가 mse : {:.3f}'.format(mse))
    #print('test dataset으로 모델 평가 loss : {:.3f}'.format(loss))  # 오차
    mae = 'mean_absolute_error 오차 : {:.3f}'.format(mae)
    mse = 'mean_squared_error 오차 : {:.3f}'.format(mse)
    loss = 'loss 오차 : {:.3f}'.format(loss)
    
    # 예측
    test_pred = model.predict(st_test_data).flatten()
    x = np.round(test_pred.tolist(), 2)
    #print("예측 값 :\n", np.round(test_pred.tolist(), 2))
    #print("실제 값 :\n", y)
    
    # 데이터 분포와 모델에 의한 선형 회귀선 시각화
    fig = px.scatter(x=test_labels, y=test_pred, width=700, height=700)
    fig.update_layout(title="",
                  xaxis_title="True value[발표수]",
                  yaxis_title="predict value[발표수]",
                  paper_bgcolor='rgba(255,255,255,0.5)',plot_bgcolor='rgba(255,255,255,0.5)',
                  font=dict(family='Franklin Gothic', color='#000', size=17))
    plot_div1 = plot(fig, output_type='div')
    
    # 오차 분포 확인 (정규성 : 잔차항이 정규분포를 따르는지 확인)
    err = test_pred
    fig = px.histogram(err, width=700, height=700, title='error[발표수]')
    fig.update_layout(title="",
                  xaxis_title="predict error[발표수]",
                  yaxis_title="",
                  paper_bgcolor='rgba(255,255,255,0.5)',plot_bgcolor='rgba(255,255,255,0.5)',
                  font=dict(family='Franklin Gothic', color='#000', size=17))
    plot_div2 = plot(fig,output_type='div')
    
    x_train, x_test, y_train, y_test = train_test_split(xt, yt, test_size=0.3, random_state=123)
    
    # 모델 구성
    inputs = Input(shape=(7, ))
    output1 = Dense(64, activation='relu')(inputs)
    output2 = Dense(32, activation='relu')(output1)
    output3 = Dense(16, activation='relu')(output2)
    output4 = Dense(1, activation='sigmoid')(output3)
    model = Model(inputs, output4)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(x_train, y_train, epochs=500, batch_size=64, verbose=1)
    
    scores = model.evaluate(x_test, y_test)
    
    print('%s : %.2f%%'%(model.metrics_names[1], scores[1] * 100))
    aww = scores[1] * 100
    acc = '%.2f%%'%aww
    #print(x_test[:1])
    pred = model.predict(xt)
    xh = np.where(pred.flatten() > 0.5, 1, 0)
    #print('예측 결과 :\n', np.where(pred.flatten() > 0.5, 1, 0))    # pred 값이 0.5보다 크면 1, 아니면 0을 보여준다
    #print('실제 결과 :\n', yh)
    
    fig = px.line(history.history, y='loss', width=700, height=700)
    fig.update_layout(title="",
                  xaxis_title="epoch",
                  yaxis_title="",
                  paper_bgcolor='rgba(255,255,255,0.5)',plot_bgcolor='rgba(255,255,255,0.5)',
                  font=dict(family='Franklin Gothic', color='#000', size=17))
    plot_div3 = plot(fig, output_type='div')
    xh = xh.tolist()
    
    x = x[:20]
    y = y[:20]
    xh = xh[:20]
    yh = yh[:20]
    
    x = json.dumps(x, cls=NumpyEncoder)
    y = json.dumps(y, cls=NumpyEncoder)
    xh = json.dumps(xh, cls=NumpyEncoder)
    yh = json.dumps(yh, cls=NumpyEncoder)
   
    data={'ks_graph_mae':plot_div01,'ks_graph_mse':plot_div02,'ks_graph_linear':plot_div1,'ks_graph_err':plot_div2,'ks_graph_logit':plot_div3, 'loss':loss, 'mae':mae, 'mse':mse,'acc':acc, 'x':x, 'y':y, 'xh':xh, 'yh':yh}
    return HttpResponse(json.dumps(data),content_type='application/json')








def Randomajax(request):
    # 모델 평가 생성
    from sklearn.ensemble import RandomForestRegressor
    data = pd.read_csv('https://raw.githubusercontent.com/pyh3887/Django_final_project/master/student.csv', encoding='euc-kr')
    
    df2 = pd.DataFrame({"결석일수":data['결석일수'],"발표수":data['발표수'],"새공지사항확인수":data['새공지사항확인수'],"토론참여수":data['토론참여수']})
    df2['결석일수'] = data['결석일수'].map({'Under-7':0,'Above-7':1})
    
    
    x = df2[['발표수','토론참여수','새공지사항확인수']].values # 2차원
    y = df2[['결석일수']].values # 1차원
    
    model = RandomForestRegressor(n_estimators=1000, criterion='mse').fit(x,y)
    #model = LogisticRegression()
    #model.fit(train_features, train_labels)
    
    # 모델 평가 생성 - > 정확도 분석
    print('예측값 : ' ,model.predict(x)[:10])
    print('실제값 : ', y[:10].ravel())
    a = model.predict(x)
    xh = np.where(a.flatten() > 0.5,1,0)
    # 학습 세트에서의 정확도
    print(xh.shape)
    print(y.shape)
    print(accuracy_score(y,xh)) # 72.7% 맞춤
    
    new_x = [[0,50,10]]
    print('새로운 값으로 예측 : ' ,model.predict(new_x))
    aa = np.where(model.predict(new_x).flatten() > 0.5,1,0)
    
    if aa == 1 :
        
        anal = '결석일수가 7일 이상으로 예측됩니다.'    
    else:
        anal = '결석일수가 7일 이하로 예측됩니다.'
        
    datas = {'anal':anal}

    return HttpResponse(json.dumps(datas),content_type='application/json')

def tensorFunc(request):
    #tensorflow
    
    
    data2 = pd.read_csv('https://raw.githubusercontent.com/pyh3887/Django_final_project/master/education.csv',encoding='euc-kr')
   
   
    print(data2)
#     label = LabelEncoder()
#     Cat_Colums = data2.dtypes.pipe(lambda Features: Features[Features=='object']).index
#           
#     for col in Cat_Colums:
#         data2[col] = label.fit_transform(data2[col])
#      
    data2.loc[data2['성적'] == 'H','성적'] = 2
    data2.loc[data2['성적'] == 'M','성적'] = 1
    data2.loc[data2['성적'] == 'L','성적'] = 0
     
    
    
    print(data2)
    #x_df = data2[['발표수','과정반복수','새공지사항확인수','토론참여수']]
    #x_df = (x_df- x_df.mean())/x_df.std()
    #print(x_df)
    #dataset2 = x_df.values
    dataset = data2.values
    print(dataset)
    # x = dataset2[:,0:4]# feature
    x = dataset[:,9:13].astype(float)# feature 
    y = dataset[:,-1]
    #print(y)
    nb_classes = 3  #label 7가지
    y_one_hot = to_categorical(y,num_classes= nb_classes)
    print(x)
    print(y_one_hot[:3])
    model = Sequential()
    
    model.add(Dense(64,input_shape=(4,),activation='relu')) # 입력데이터(노드) 보단 출력데이터(유닛)이 더 많도록하자(병목현상 방지)
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))      
    model.add(Dense(3,activation='softmax')) # softmax 
    model.summary()
    model.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics='accuracy')
    
    early_stop = EarlyStopping(monitor='loss',patience=10)
    history = model.fit(x,y_one_hot,epochs=500, batch_size = 100, validation_split=0.3, verbose=2,callbacks=[early_stop]) # 딥러닝 시작 
    scores = model.evaluate(x,y_one_hot)

    
    #print('eval : ' , history.history['accuracy'][-1])
    accuracy = round(history.history['accuracy'][-1],2) * 100
     
    # 새로운 데이터로 
    # M 0,0,1 L 0,1,0 H 1,0,0    
      
    new_data = np.array([[20,14,12,19]])
    
    print(np.argmax(model.predict(new_data)))
  
    if np.argmax(model.predict(new_data)) == 2:
        a = '성적이  상위권으로 예측됩니다.'
          
 
    elif np.argmax(model.predict(new_data)) == 1:
        a = '성적이 중위권으로 예측됩니다.'
 
    else :
        a = '성적이 하위권 으로 예측됩니다.'
     
    print(a)
    loss = history.history['loss']
    epochs = range(1,len(loss)+1)         

  
    
    # Create traces
    yh_fig3 = go.Figure()
 
    yh_fig3.add_trace(go.Scatter(x=history.epoch, y=history.history['loss'],
                        mode='lines',
                        name='loss'))
    yh_fig3.add_trace(go.Scatter(x=history.epoch, y=history.history['val_loss'],
                        mode='lines+markers',
                        name='val_loss'))
    yh_fig3.add_trace(go.Scatter(x=history.epoch, y=history.history['accuracy'],
                        mode='markers', name='accuracy'))
    yh_fig3.add_trace(go.Scatter(x=history.epoch, y=history.history['val_accuracy'],
                        mode='markers', name='val_accuracy'))
    yh_fig3.update_layout(title='Tensorflow 정확도와 손실값')
    
    yh_grap3 = plot(yh_fig3,output_type='div')   
    
    
    datas = {'a': a, 'yh_grap3':yh_grap3,'accuracy':accuracy} 
    
    
    
    
    
        
    #print(datas)
    return HttpResponse(json.dumps(datas),content_type='application/json')