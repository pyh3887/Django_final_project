from django.shortcuts import render
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.utils import to_categorical

import os
plt.rc('font',family='malgun gothic')
# Create your views here.
def mainFunc(request):
    
    data = pd.read_csv('https://raw.githubusercontent.com/pyh3887/Django_final_project/master/education.csv',encoding='euc-kr')
    
    print(data)
    label = LabelEncoder()
    Cat_Colums = data.dtypes.pipe(lambda Features: Features[Features=='object']).index
     
    
    for col in Cat_Colums:
        data[col] = label.fit_transform(data[col])
    
   
    
    x = data[['성별','국적','교육단계','학년','학급','전공','학기','발표수','과정반복수','새공지사항확인수','토론참여수','결석일수','부모의학교만족도']]
    y = data['성적']
    
    print(x)
    print(y)
    
    train_x,test_x,train_y,test_y = train_test_split(x,y)
        
    model = XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=100,seed=10)
    fit_model = model.fit(train_x,train_y)

    pred = fit_model.predict(test_x)

    #print('예측값 : ', pred[:5])
    #print('실제값 : ', np.array(test_y[:5]))
    from sklearn.metrics import accuracy_score
    #print('분류 정확도 : ', accuracy_score(test_y, pred))
    
    #print('특성 중요도 :\n{}'.format(model.feature_importances_))
    plot_importance(model)
    plt.xlabel('안녕')
    fig = plt.gcf()
    fig.savefig(os.path.dirname(os.path.realpath(__file__)) + '\\static\\bbb.jpg')
  
    data2 = pd.read_csv('https://raw.githubusercontent.com/pyh3887/Django_final_project/master/education.csv',encoding='euc-kr')
    
    
    print(data2)
    label = LabelEncoder()
    Cat_Colums = data2.dtypes.pipe(lambda Features: Features[Features=='object']).index
      
    for col in Cat_Colums:
        data2[col] = label.fit_transform(data2[col])
    
#     data2.loc[data2['성적'] == 'H','성적'] = 2
#     data2.loc[data2['성적'] == 'M','성적'] = 1
#     data2.loc[data2['성적'] == 'L','성적'] = 0
    
    print(data2)
    dataset = data2.values
    print(dataset)
    x = dataset[:,9:13] # feature
    
    y = dataset[:,-1]
    #print(y)
    nb_classes = 3  #label 7가지
    y_one_hot = to_categorical(y,num_classes= nb_classes)
    
    model = Sequential()

    model.add(Dense(32,input_shape=(3,),activation='relu')) # 입력데이터(노드) 보단 출력데이터(유닛)이 더 많도록하자(병목현상 방지)
    model.add(Dense(16, activation='relu')) 
    model.add(Dense(nb_classes,activation='softmax')) #마지막 값은 sigmoid 
    model.summary()
    model.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics='accuracy')

    history = model.fit(x,y_one_hot,epochs=100,batch_size=64, verbose=1) # 딥러닝 시작 

    scores = model.evaluate(x,y) 
    print('eval : ' , model.evaluate(x,y_one_hot))
    
    # 새로운 데이터로 
    print(x[:1])
    #new_data = np.array([[1.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,12.,0.,0.,0.]]) 
    #print(np.argmax(model.predict(new_data))) # 5번 동물로 분류 

    
    
    
    
    return render(request,'ax.html')
