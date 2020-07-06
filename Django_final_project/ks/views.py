from django.shortcuts import render
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import os
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from plotly.offline import plot
from tensorflow.keras import layers
from xgboost.callback import early_stop
from plotly.subplots import make_subplots

plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False

# Create your views here.

# 각 칼럼 별로 손 든 횟수와 상관 계수를 구해 상관이 있는지를 구한 다음 예측을 해본다. (corr_ex2)
def mainFunc(request):
    data = pd.read_csv('https://raw.githubusercontent.com/pyh3887/Django_final_project/master/student.csv', encoding='euc-kr')
# '성별', '국적', '출생지', '교육단계', '학년', '학급', '전공', '학기', '담당부모', '발표수', '과정반복수', '새공지사항확인수', '토론참여수', '부모의학교만족도', '결석일수', '성적'
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

#     이전에 저장해서 쓰던 방식
#     plt.figure(figsize=(9, 9))
#     sns.heatmap(df.corr(), annot=True, fmt = '.2f', linewidths=.5, cmap='RdBu_r')
#     fig = plt.gcf()
#     fig.savefig(os.path.dirname(os.path.realpath(__file__)) + '\\static\\images\\heatmap.png')
#     plt.clf()

    fig = px.imshow(df.corr(),
                    x=['성별', '국적', '출생지', '교육단계', '학년', '학급', '전공', '학기', '담당부모', '발표수', '과정반복수', '새공지사항확인수', '토론참여수', '부모의학교만족도', '결석일수', '성적'],
                    y=['성별', '국적', '출생지', '교육단계', '학년', '학급', '전공', '학기', '담당부모', '발표수', '과정반복수', '새공지사항확인수', '토론참여수', '부모의학교만족도', '결석일수', '성적'],
                    width=900, height=900, color_continuous_scale='RdBu_r'
               )
    plot_div = plot(fig,output_type='div')
    
    # 발표 수와 상관관계의 값이 0.3 이상인 것은 담당부모, 과정반복수, 새공지사항확인수, 토론참여수, 부모의학교만족도, 결석일수, 성적이 있다.
    # 그러면 이 칼럼들로 담당부모를 예측해 보겠다.
#     print(df['담당부모'])
    xhas = df.loc[:, ['담당부모', '발표수', '과정반복수', '새공지사항확인수', '토론참여수', '부모의학교만족도', '결석일수', '성적']]
    yhas = df.loc[:, ['발표수']]
    y = df.발표수.tolist()
    
#     xdata = xhas.values
#     ydata = yhas.values
#     print(xdata[:2], xdata.shape)
#     print(ydata[:2], ydata.shape)
#     
#     model = Sequential()
#     model.add(Dense(64, input_shape=(7, ), activation='relu'))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(2, activation='linear'))
#     print(model.summary())
#     model.compile(optimizer='adam', loss='mse', metrics=['acc'])
#     
#     model.fit(xdata, ydata, epochs=100, batch_size=10, verbose=0)
#     model_eval = model.evaluate(xdata, ydata)
#     print('model_eval :', model_eval)
#     new_x = [[1, 80, 50, 100, 1, 10, 3]]
#     print('예측 값 :', model.predict(xdata))
#     print('실제 값 :', ydata)

    train_dataset = xhas.sample(frac=0.7, random_state=0)
    test_dataset = xhas.drop(train_dataset.index)
    print(train_dataset.shape)      # (336, 8)
    print(test_dataset.shape)       # (144, 8)
    
    train_stat = train_dataset.describe()
    train_stat.pop('발표수')
    train_stat = train_stat.transpose()
    print(train_stat)       # std열이 표준 편차이다.
    
    # label
    train_labels = train_dataset.pop('발표수')
    print(train_labels[:2])
    test_labels = test_dataset.pop('발표수')
    print(test_labels[:2])
    
    def st_func(x): # 표준화 처리 함수    (요소 값 - 평균) / 표준 편차
        return ((x - train_stat['mean']) / train_stat['std'])
    
    print('st_func(10) :\n', st_func(10))
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
    print(model.summary())
    
    # fit() 전에 모델을 실행해 볼 수도 있다.
    print(model.predict(st_train_data[:1]))     # 결과는 신경쓰지 않는다.
    
    # 모델 훈련
    epochs = 5000
    
    # 학습 조기 종료
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    
    history = model.fit(st_train_data,train_labels, epochs=epochs, validation_split=0.2, verbose=1, callbacks=[early_stop]) 
    # 원래 데이터셋이 1000개 이고, fit 함수의 validation_split = 0.2 로 하면, 
    # training dataset 은 800개로 하여, training 시키고,나머지 200개는 test dataset 으로 사용하여, 모델을 평가하게 된다.
    df = pd.DataFrame(history.history)
    print(df.head())
    print(df.columns)
    # ['loss', 'mean_squared_error', 'mean_absolute_error',
    #   'val_loss', 'val_mean_squared_error', 'val_mean_absolute_error']    # validation_split을 썻기 때문에 나오는 것이다. 안쓰면 위에 3개만 나온다.
    
#     def plot_history(history):
#         hist = pd.DataFrame(history.history)
#         hist['epoch'] = history.epoch
#         print(hist)
#         print(hist.columns)
#         df = px.data.gapminder().query("continent=='Oceania'")
#         print(df)
#         
#         plt.figure(figsize=(8,12))
#         
#         plt.subplot(2,1,1)
#         plt.xlabel('Epoch')
#         plt.ylabel('Mean Abs Error [발표수]')
#         plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
#         plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label ='Val Error')
#         plt.legend()
#         
#         plt.subplot(2,1,2)
#         plt.xlabel('Epoch')
#         plt.ylabel('Mean Square Error [$발표수^2$]')
#         plt.plot(hist['epoch'], hist['mean_squared_error'],label='Train Error')
#         plt.plot(hist['epoch'], hist['val_mean_squared_error'],label = 'Val Error')
#         plt.legend()
#     
#     plot_history(history)
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['mean_absolute_error'], name='Train Error',
                             line=dict(color='firebrick', width=4)))
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_mean_absolute_error'], name = 'Val Error',
                             line=dict(color='royalblue', width=4)))
    
    fig.update_layout(title='plot_history',
               xaxis_title='Epoch',
               yaxis_title='Mean Abs Error [발표수]',
               width=600, height=600)
    plot_div01 = plot(fig, output_type='div')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['mean_squared_error'], name='Train Error',
                             line=dict(color='firebrick', width=4)))
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_mean_squared_error'], name = 'Val Error',
                             line=dict(color='royalblue', width=4)))
    
    fig.update_layout(title='plot_history',
               xaxis_title='Epoch',
               yaxis_title='Mean Square Error [발표수]',
               width=600, height=600)
    plot_div02 = plot(fig, output_type='div')
    
    # 모델 평가
    loss, mae, mse = model.evaluate(st_test_data, test_labels)
    print('test dataset으로 모델 평가 mae : {:5.3f}'.format(mae))
    print('test dataset으로 모델 평가 mse : {:5.3f}'.format(mse))
    print('test dataset으로 모델 평가 loss : {:5.3f}'.format(loss))  # 오차
    mae = 'mean_absolute_error 오차 : {:5.3f}'.format(mae)
    mse = 'mean_squared_error 오차 : {:5.3f}'.format(mse)
    loss = 'loss 오차 : {:5.3f}'.format(loss)
    
    # 예측
    test_pred = model.predict(st_test_data).flatten()
    print("예측 값 :\n", test_pred)
    print("실제 값 :\n", y)
    
    # 데이터 분포와 모델에 의한 선형 회귀선 시각화
    fig = px.scatter(x=test_labels, y=test_pred, width=900, height=900)
    plot_div1 = plot(fig, output_type='div')
    
    # 오차 분포 확인 (정규성 : 잔차항이 정규분포를 따르는지 확인)
    err = test_pred
    fig = px.histogram(err, width=900, height=900, title='error[발표수]')
    plot_div2 = plot(fig,output_type='div')
    
    return render(request,'pearson.html',
                  context={'plot_div':plot_div, 'plot_div01':plot_div01, 'plot_div02':plot_div02, 'plot_div1':plot_div1, 'plot_div2':plot_div2, 'loss':loss, 'mae':mae, 'mse':mse, 'test_pred':test_pred, 'y':y}
                  )
