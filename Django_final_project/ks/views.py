from django.shortcuts import render
import pandas as pd

# Create your views here.


def mainFunc(request):
    df2018 = pd.read_csv('C:/Users/KITCOOP/git/Django_final_project/Django_final_project/ks/static/2018.csv', encoding='euc-kr')
    df2017 = pd.read_csv('C:/Users/KITCOOP/git/Django_final_project/Django_final_project/ks/static/2017.csv', encoding='euc-kr')
    df2016 = pd.read_csv('C:/Users/KITCOOP/git/Django_final_project/Django_final_project/ks/static/2016.csv', encoding='euc-kr')
    df2015 = pd.read_csv('C:/Users/KITCOOP/git/Django_final_project/Django_final_project/ks/static/2015.csv', encoding='euc-kr')
#     print(df2015)
    
    chi2015 = df2015['맞벌이여부_유아동,청소년']
    print(chi2015)
    return render(request,'aa.html')
