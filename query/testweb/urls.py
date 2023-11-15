"""testweb URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
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
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.urls import path
from django.urls import include
import json

question="暂无问题"

def hello(request):
    text=request.GET.get('login_status')
    return render(request, 'hello.html')
def login(request):
    if 'GET' == request.method:
        return render(request, 'login.html')
    else:
        name = request.POST.get('username')
        pwd = request.POST.get('password')
        if '1' == name :
            return redirect('/ivr')
        else:
            return render(request, 'login.html', {'login_status':'texta'},)

def test(request):
    return render(request,'test.html')


def login2(request):

        if 'zackm' == name and '123456' == pwd:

            return render(request, 'login.html', {'login_status':'textabb'})

def IVRtree(request):
    return render(request,'IVRtree.html')

def suiyi(request):
    return render(request,'suiyi.html')

def click(request):
    return render(request, 'login.html', {'login_status':'click'},)

def graphSearch():
    print("graph search success")

def Inputsearch(request):
    global question
    question=request.POST.get('content')
    graphSearch()
    if question.find("调用")!=-1 :
        return HttpResponse("/ivr")
    elif question.find("功能")!=-1 or question.find("信息")!=-1 or question.find("字符串")!=-1:
        return HttpResponse('/answer')
    else :
        return HttpResponse("none")

def Query(request):
    query(request)

def answer(request):
    return render(request, "../templates/answer.html")

def answer1(request):
    return render(request, "answer1.html")

def testanswer(request):
    return render(request, "testanswer.html")

def getAnswerText(request):
    return HttpResponse(question)

def oversummary(request):
    return render(request, "../static/overview-summary.html")

def overframe(request):
    return render(request, "../static/overview-frame.html")

def reference(request):
    return render(request, "reference.html")

def reference1(request):
    return render(request, "reference1.html")

def sourcecode(request):
    return render(request, "sourcecode.html")

def sourcecode1(request):
    return render(request, "sourcecode1.html")

def code(request):
    return render(request, "code.html")

def cassert(request):
    return render(request, "cassert.html")

urlpatterns = [
    path('admin/', admin.site.urls),
    path('hello/', hello ),
    path('login/',login),
    path('ivr/',IVRtree),
    path('test',test),
    path('suiyi', suiyi),
    path('click', click),
    path('inputSearch', Inputsearch),
    path('answer/', answer),
    path('answer1/', answer1),
    path('testanswer/', testanswer),
    path('getAnswerText/', getAnswerText),
    path('overview-summary.html',oversummary),
    path('overview-frame.html',overframe),
    path('reference/',reference),
    path('reference1/',reference1),
    path('sourcecode/',sourcecode),
    path('sourcecode1/',sourcecode1),
    path('code/',code),
    path('cassert/',cassert),
]
