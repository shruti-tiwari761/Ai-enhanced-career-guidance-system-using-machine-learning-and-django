from django.urls import path
from home import views

urlpatterns=[
    path('',views.index,name='home'),
    path('about',views.about,name='about'),
    path('contact/', views.contact, name='contact'),
    path('prediction',views.prediction,name='prediction'),
    path('login',views.loginpage,name='login'),
    path('signup',views.signuppage,name='signup'),
    path('class',views.classpage,name='class'),
    path('class9thand10th',views.class9thand10th,name='class'),
    path('Mathsprediction',views.Math,name='class'),
    path('Scienceprediction',views.Science,name='class'),
    path('Commerceprediction',views.Commerce,name='class'),
    path('Artsprediction',views.Arts,name='class'),
    path('chat',views.chat,name='chat'),
    path('profile/', views.profile, name='profile'),
]