from django.shortcuts import render

# Create your views here.




def mainFunc(request):
    print('aa')  
    return render(request,'aa.html')
