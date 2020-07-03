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
from tensorflow.keras.callbacks import  EarlyStopping


import os
from tensorflow.keras import layers
plt.rc('font',family='malgun gothic')
# Create your views here.
def mainFunc(request):
    
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
    fig.savefig(os.path.dirname(os.path.realpath(__file__)) + '\\static\\bbb.png')
  
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
    
    #early_stop = EarlyStopping(monitor='loss',patience=100, baseline=0.4)
    history = model.fit(x,y_one_hot,epochs=500, batch_size = 100, validation_split=0.3, verbose=2) # 딥러닝 시작 
    scores = model.evaluate(x,y_one_hot)
    
    print('eval : ' , scores)
     
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
         
            
    
    
    
    
    
    return render(request,'ax.html',{'a':a})
