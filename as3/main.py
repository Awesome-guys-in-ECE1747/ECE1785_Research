# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
from sklearn.linear_model import LinearRegression as lr
def data_cleaning(data):
    # print()
    # Use a breakpoint in the code line below to debug your script.
    # print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    data1=data.loc[:,['OSS.HIRING','EMPLOYER.POLICY.APPLICATIONS','AGE']].dropna()
    # print(data1.describe())
    # print("which has Nan?\n", data1.isnull().sum(), "\n")
    hiring={'Very important':1,'Somewhat important':2,'Not at all important':3,'Not too important':4,'Not applicable-I hadn\'t made any contributions when I got this job.':5}
    age={'17 or younger':1,'18 to 24 years':2,'25 to 34 years':3,'35 to 44 years':4,'45 to 54 years':5,'55 to 64 years':6,'65 years or older':7}
    policy={'Use of open source applications is encouraged.':1,'Use of open source applications is acceptable if it is the most appropriate tool.':2,'My employer doesn\'t have a clear policy on this.':3,'Not applicable':4,'I\'m not sure.':5,'Use of open source applications is rarely, if ever, permitted.':6}
    # policy={'I am free to contribute without asking for permission.':1,'My employer doesn\'t have a clear policy on this.':2,'I am permitted to contribute to open source, but need to ask for permission.':3,'I\'m not sure.':4,'Not applicable':5,'I am not permitted to contribute to open source at all.':6}
    job={'Yes, directly-  some or all of my work duties include contributing to open source projects.':1,'Yes, indirectly- I contribute to open source in carrying out my work duties, but I am not required or expected to do so.':2,'No.':3}
    data1['OSS.HIRING']=data1['OSS.HIRING'].map(hiring)
    data1['EMPLOYER.POLICY.APPLICATIONS'] = data1['EMPLOYER.POLICY.APPLICATIONS'].map(policy)
    data1['AGE'] = data1['AGE'].map(age)
    data2=data.loc[:,['OSS.AS.JOB','EMPLOYER.POLICY.APPLICATIONS']].dropna()
    data2['OSS.AS.JOB']=data2['OSS.AS.JOB'].map(job)
    data2['EMPLOYER.POLICY.APPLICATIONS'] = data2['EMPLOYER.POLICY.APPLICATIONS'].map(policy)
    # print("which has Nan?\n", data2.isnull().sum(), "\n")
    return data1,data2

def data1_analysis(data):
    # print(data.describe())
    # print(data.head(5))
    lrmodel=lr()
    # print("which has Nan?\n", data.isnull().sum(), "\n")
    data[['AGE','EMPLOYER.POLICY.APPLICATIONS','OSS.HIRING']].corr()
    x=data[['AGE','EMPLOYER.POLICY.APPLICATIONS']]
    y=data[['OSS.HIRING']]
    lrmodel.fit(x,y)
    print(lrmodel.coef_)
    for i in range(7):
        for j in range(6):
            print(i+1,j+1,lrmodel.predict([[i+1,j+1]])[0][0])
    # print(lrmodel.predict([[2,1]]))

def data2_analysis(data):
    lrmodel=lr()
    data[['EMPLOYER.POLICY.APPLICATIONS','OSS.AS.JOB']].corr()
    x=data[['EMPLOYER.POLICY.APPLICATIONS']]
    y=data[['OSS.AS.JOB']]
    lrmodel.fit(x,y)
    print(lrmodel.coef_)
    for i in range(6):
        print(i + 1, lrmodel.predict([[i + 1]])[0][0])
    print('1')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    data=pd.read_csv('./survey_data.csv')
    data1,data2=data_cleaning(data)
    data1_analysis(data1)
    data2_analysis(data2)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
