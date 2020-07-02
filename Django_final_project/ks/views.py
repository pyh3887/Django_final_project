from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False

# Create your views here.

# 각 칼럼 별로 손 든 횟수와 상관 계수를 구해 상관이 있는지를 구한 다음 예측을 해본다. (corr_ex2)
def mainFunc(request):
    data = pd.read_csv('https://raw.githubusercontent.com/pyh3887/Django_final_project/master/student.csv', encoding='euc-kr')
# '성별', '국적', '출생지', '교육단계', '학년', '학급', '전공', '학기', '담당부모', '발표수', '과정반복수', '새공지사항확인수', '토론참여수', '부모의학교만족도', '결석일수', '성적'
    df = data
    # LabelEncoder
    le = LabelEncoder()
    # apply "le.fit_transform"
    df = df.apply(le.fit_transform)
    df.loc[df['성적'] == 0, '성적'] = 3
    df.loc[df['성적'] == 2, '성적'] = 2
    df.loc[df['성적'] == 1, '성적'] = 1
    df.loc[df['성별'] == 1, '성별'] = 2
    df.loc[df['성별'] == 2, '성별'] = 1
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

    plt.figure(figsize=(9, 9))
    sns.heatmap(df.corr(), annot=True, fmt = '.2f', linewidths=.5, cmap='RdBu_r')
    fig = plt.gcf()
    fig.savefig(os.path.dirname(os.path.realpath(__file__)) + '\\static\\images\\heatmap.png')
    plt.clf()
    # 발표 수와 상관관계의 값이 0.3 이상인 것은 담당부모, 과정반복수, 새공지사항확인수, 토론참여수, 부모의학교만족도, 결석일수가 있다.
    # 그러면 이 칼럼들로 발표수를 예측해 보겠다.
    print(df['담당부모'])
    has = df.iloc[:, 9] + df.iloc[:, 11:15]
#     print(has)
#     xdata = np.random.random((1000, 12))
#     ydata = np.random.randint(10, size=(1000, 1))
#     ydata = to_categorical(ydata, num_classes=10)
#     print(xdata[:2], xdata.shape)
#     print(ydata[:2], ydata.shape)
#     
#     model = Sequential()
#     model.add(Dense(100, input_shape=(12, ), activation='relu'))
#     model.add(Dense(50, activation='relu'))
#     model.add(Dense(10, activation='softmax'))
#     print(model.summary())
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
#     
#     hist = model.fit(xdata, ydata, epochs=500, batch_size=32, verbose=2)
#     model_eval = model.evaluate(xdata, ydata)
#     print('model_eval :', model_eval)
#     
#     print('예측 값 :', [np.argmax(i) for i in (model.predict(xdata[:5]))])     # 가장 큰 값을 찾기 위해서 np.argmax를 사용한다.
#     print('실제 값 :', [np.argmax(i) for i in ydata[:5]])
    
    return render(request,'aa.html')
