from django.shortcuts import render

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# Create your views here.
def mainFunc(request):
    data = pd.read_csv('https://raw.githubusercontent.com/pyh3887/Django_final_project/master/student.csv', encoding='euc-kr')

    # 데이터 상위 5개 행 읽기
    print(data.head())
    # 데이터의 컬럼별 요약 통계량 확인
    print(data.describe())
    # 데이터의 행과 열의 개수
    print(data.shape)
    # 결측값 확인
    print(data.isnull().sum())
    
    
    # 국적 인원 분포도 (1)
    print(data['국적'].value_counts())
    # 국적 인원 분포도 퍼센트로 막대그래프 출력 (2)
    print('인원수\n',data.국적.value_counts(normalize=False))
    plt.subplot(1, 2, 1)
    data.국적.value_counts(normalize=False).plot(kind='pie')
    
    # 국적별 성적 분포도
    print('인원수\n',data.성적.value_counts(normalize=False))
    plt.subplot(1, 2, 1)
    data.성적.value_counts(normalize=False).plot(kind='pie')
    
    flg = plt.gcf()
    
    flg.savefig(os.path.dirname(os.path.realpath(__file__)) + '\\static\\class_cuntry.jpg')
    return render(request,'grade_cuntry_pie.html')
