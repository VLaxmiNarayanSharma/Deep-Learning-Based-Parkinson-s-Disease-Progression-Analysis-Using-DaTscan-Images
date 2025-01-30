from django.contrib import messages
from django.shortcuts import render, HttpResponse
from django.conf import settings
import os
from .forms import UserRegistrationForm
from .models import UserRegistrationModel


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})


def user_view_dataset(request):
    path = os.path.join(settings.MEDIA_ROOT, 'parkinsons.csv')
    import pandas as pd
    df = pd.read_csv(path)
    df = df.drop(['name', 'MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ','Jitter:DDP','spread1','spread2','MDVP:Shimmer'], axis=1)
    df = df.to_html
    return render(request, 'users/view_data.html', {'df': df})


def user_model_evaluations(request):
    from .utility.parkinson_utility import start_models
    result = start_models()
    print(result)
    return render(request, 'users/model_results.html',result)


def user_predict_form(request):
    if request.method == 'POST':
        nhr = float(request.POST.get('NHR'))
        hnr = float(request.POST.get('HNR'))
        rpde = float(request.POST.get('RPDE'))
        dfa = float(request.POST.get('DFA'))
        ppe = float(request.POST.get('PPE'))

        test_data = [nhr, hnr,rpde,dfa, ppe]
        from .utility import process_user_input
        test_pred = process_user_input.get_result(test_data)
        print("Test Result is:", test_pred)
        if test_pred[0] == 0:
            rslt = False
        else:
            rslt = True
        return render(request, "users/parkinson_form.html", {"test_data": test_data, "result": rslt})
    else:

        return render(request, 'users/parkinson_form.html', {})
