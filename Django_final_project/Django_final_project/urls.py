"""Django_final_project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.urls.conf import include
from yh import views

urlpatterns = [ 
    path('admin/', admin.site.urls),
    path('', admin.site.urls),
    path('ks/',include('ks.urls')),
    path('yj/',include('yj.urls')),
    path('yh/',include('yh.urls')),
    path('cg/',include('cg.urls')),
    path('show',views.tensorFunc),
    path('show2',views.Randomajax),  
    path('show3',views.ksajax)
]
