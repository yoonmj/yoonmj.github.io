---
layout: single
title: "로지스틱 회귀 분석 정리"
published: true
---

**Python을 사용한 로지스틱 회귀 분석 분류기 자습서**
=============

[Logistic Regression Classifier](https://www.kaggle.com/code/prashant111/logistic-regression-classifier-tutorial#5.-Import-libraries-)
다음 링크를 보고 따라한 글입니다. 부족한 부분은 링크를 참고하기

1. 로지스틱 회귀 분석 소개

데이터 과학자들이 새로운 분류 문제를 발견할 수 있는 경우, 가장 먼저 떠오르는
알고리즘은 로지스틱 회귀 분석입니다. 이것은 개별 클래스 집합에 대한 관찰을 예측하
는 데 사용되는 지도 학습 분류 알고리즘입니다. 실제로 관측치를 여러 범주로 분류하는
데 사용됩니다. 따라서, 그것의 출력은 본질적으로 별개입니다. 로지스틱 회귀 분석을 
로짓 회귀 분석이라고도 합니다. 분류 문제를 해결하는 데 사용되는 가장 단순하고 
간단하며 다용도의 분류 알고리즘 중 하나입니다.


2. 로지스틱 회귀 직관

통계학에서 로지스틱 회귀 모형은 주로 분류 목적으로 사용되는 널리 사용되는 통계 모형입니다. 즉, 관측치 집합이 주어지면 로지스틱 회귀 알고리즘을 사용하여 관측치를 두 개 이상의 이산 클래스로 분류할 수 있습니다. 따라서 대상 변수는 본질적으로 이산적입니다.

선형 방정식 구현
로지스틱 회귀 분석 알고리즘은 반응 값을 예측하기 위해 독립 변수 또는 설명 변수가 있는 선형 방정식을 구현하는 방식으로 작동합니다. 예를 들어, 우리는 공부한 시간의 수와 시험에 합격할 확률의 예를 고려합니다. 여기서 공부한 시간 수는 설명 변수이며 x1로 표시됩니다. 합격 확률은 반응 변수 또는 목표 변수이며 z로 표시됩니다.

만약 우리가 하나의 설명 변수(x1)와 하나의 반응 변수(z)를 가지고 있다면, 선형 방정식은 다음과 같은 방정식으로 수학적으로 주어질 것입니다
z = β0 + β1x1    
여기서 계수 β0과 β1은 모형의 모수입니다.
설명 변수가 여러 개인 경우, 위의 방정식은 다음과 같이 확장될 수 있습니다
z = β0 + β1x1 + β2x2+….+ βnxn

여기서 계수 β0, β1, β2 및 βn은 모델의 매개변수입니다.
따라서 예측 반응 값은 위의 방정식에 의해 주어지며 z로 표시됩니다.


시그모이드 함수
z로 표시된 이 예측 반응 값은 0과 1 사이에 있는 확률 값으로 변환됩니다. 우리는 예측 값을 확률 값에 매핑하기 위해 시그모이드 함수를 사용합니다. 그런 다음 이 시그모이드 함수는 실제 값을 0과 1 사이의 확률 값으로 매핑합니다.

기계 학습에서 시그모이드 함수는 예측을 확률에 매핑하는 데 사용됩니다. 시그모이드 함수는 S자형 곡선을 가지고 있습니다. 그것은 시그모이드 곡선이라고도 불립니다.

Sigmoid 함수는 로지스틱 함수의 특수한 경우입니다. 그것은 다음과 같은 수학 공식에 의해 주어집니다.

다음 그래프로 시그모이드 함수를 그래픽으로 표현할 수 있습니다

 ![nn](./rr.png)

의사결정경계
시그모이드 함수는 0과 1 사이의 확률 값을 반환합니다. 그런 다음 이 확률 값은 "0" 또는 "1"인 이산 클래스에 매핑됩니다. 이 확률 값을 이산 클래스(통과/실패, 예/아니오, 참/거짓)에 매핑하기 위해 임계값을 선택합니다. 이 임계값을 의사결정 경계라고 합니다. 이 임계값을 초과하면 확률 값을 클래스 1에 매핑하고 클래스 0에 매핑합니다.

p ≥ 0.5 => class = 1
p < 0.5 => class = 0

일반적으로 의사 결정 경계는 0.5로 설정됩니다. 따라서 확률 값이 0.8(> 0.5)이면 이 관측치를 클래스 1에 매핑합니다. 마찬가지로 확률 값이 0.2(< 0.5)이면 이 관측치를 클래스 0에 매핑합니다. 이것은 아래 그래프에 나와 있습니다

 ![nn](./222.png)

예측하기
이제 우리는 로지스틱 회귀 분석에서 시그모이드 함수와 결정 경계에 대해 알고 있습니다. 우리는 시그모이드 함수와 결정 경계에 대한 지식을 사용하여 예측 함수를 작성할 수 있습니다. 로지스틱 회귀 분석의 예측 함수는 관측치가 양수, 예 또는 참일 확률을 반환합니다. 이를 클래스 1이라고 하며 P(클래스 = 1)로 표시합니다. 확률이 1에 가까우면 관측치가 클래스 1에 있고 그렇지 않으면 클래스 0에 있다는 것을 모형에 대해 더 확신할 수 있습니다.

3. 로지스틱 회귀 분석의 가정
로지스틱 회귀 분석모형에 필요한 몇가지 주요 가정

1.로지스틱 회귀 분석 모형에서는 종속 변수가 이항, 다항식 또는 순서형이어야 합니다.
2.관측치가 서로 독립적이어야 합니다. 따라서 관측치는 반복적인 측정에서 나와서는 안 됩니다.
3.로지스틱 회귀 분석 알고리즘에는 독립 변수 간의 다중 공선성이 거의 또는 전혀 필요하지 않습니다. 즉, 독립 변수들이 서로 너무 높은 상관 관계를 맺어서는 안 됩니다.
4.로지스틱 회귀 모형은 독립 변수와 로그 승산의 선형성을 가정합니다.
5.로지스틱 회귀 분석 모형의 성공 여부는 표본 크기에 따라 달라집니다. 일반적으로 높은 정확도를 얻으려면 큰 표본 크기가 필요합니다.

4. 로지스틱 회귀 분석의 유형

로지스틱 회귀 분석 모형은 대상 변수 범주를 기준으로 세 그룹으로 분류할 수 있습니다. 이 세 그룹은 아래에 설명되어 있습니다

1. 이항 로지스틱 회귀 분석
이항 로지스틱 회귀 분석에서 대상 변수에는 두 가지 범주가 있습니다. 범주의 일반적인 예는 예 또는 아니오, 양호 또는 불량, 참 또는 거짓, 스팸 또는 스팸 없음, 통과 또는 실패입니다.

2. 다항 로지스틱 회귀 분석
다항 로지스틱 회귀 분석에서 대상 변수에는 특정 순서가 아닌 세 개 이상의 범주가 있습니다. 따라서 세 개 이상의 공칭 범주가 있습니다. 그 예들은 사과, 망고, 오렌지 그리고 바나나와 같은 과일의 종류를 포함합니다.

3. 순서형 로지스틱 회귀 분석
순서형 로지스틱 회귀 분석에서 대상 변수에는 세 개 이상의 순서형 범주가 있습니다. 그래서, 범주와 관련된 본질적인 순서가 있습니다. 예를 들어, 학생들의 성적은 불량, 평균, 양호, 우수로 분류될 수 있습니다.

5. 라이브러리 가져오기


```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```


```python
import warnings

warnings.filterwarnings('ignore')
```

6. 데이터 집합 가져오기


```python
data = '/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv'

df = pd.read_csv(data)
```

7. 탐색적 데이터 분석


```python
df.shape
```


```python
df.head()
```


```python
col_names = df.columns

col_names
```


```python
df.drop(['RISK_MM'], axis=1, inplace=True)  #RISK_MM 변수 삭제
```


```python
df.info()
```

변수 유형
이 섹션에서는 데이터 세트를 범주형 변수와 숫자 변수로 분리합니다. 데이터 집합에는 범주형 변수와 숫자 변수가 혼합되어 있습니다. 범주형 변수에는 데이터 유형 개체가 있습니다. 숫자 변수의 데이터 유형은 float64입니다.


```python
#범주형 변수
categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)
```


```python
df[categorical].head()
```

범주형 변수 요약
날짜 변수가 있습니다. 날짜 열로 표시됩니다.
6개의 범주형 변수가 있습니다. .
두 개의 이진 범주형 변수인 RainToday와 RainTomorrow가 있습니다.
내일 비가 목표 변수입니다.


```python
df[categorical].isnull().sum()   #범주형 변수의 결측값
```


```python
cat1 = [var for var in categorical if df[var].isnull().sum()!=0]

print(df[cat1].isnull().sum())
```


```python
for var in categorical: 
    
    print(df[var].value_counts())
```


```python
for var in categorical: 
    
    print(df[var].value_counts()/np.float(len(df)))
```

날짜 변수 전처리

날짜 변수의 피쳐 엔지니어링

날짜 변수의 데이터 유형이 개체임을 알 수 있습니다. 현재 객체로 코딩된 날짜를 datetime 형식으로 구문 분석하겠습니다.


```python
df['Date'].dtypes
```


```python
df['Date'] = pd.to_datetime(df['Date'])
```


```python
df['Year'] = df['Date'].dt.year

df['Year'].head()
```


```python
df['Month'] = df['Date'].dt.month

df['Month'].head()
```


```python
df['Day'] = df['Date'].dt.day

df['Day'].head()
```


```python
df.info()
```

날짜 변수에서 추가로 세 개의 열이 생성된 것을 확인할 수 있습니다. 이제 데이터 집합에서 원래 날짜 변수를 삭제


```python
df.drop('Date', axis=1, inplace = True)
```


```python
df.head()
```

범주형 변수 탐색


```python
categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical
```

데이터 세트에 6개의 범주형 변수가 있다는 것을 알 수 있습니다. 날짜 변수가 제거되었습니다. 먼저 범주형 변수의 결측값을 확인하겠습니다.


```python
df[categorical].isnull().sum()
```

WindGustDir, WindDir9am, WindDir3pm, RainToday 변수에 결측값이 포함되어 있음을 알 수 있다

Location 변수 탐색


```python
print('Location contains', len(df.Location.unique())
```


```python
df.Location.unique()
```


```python
df.Location.value_counts()
```


```python
pd.get_dummies(df.Location, drop_first=True).head()
```

WindGustDir 변수 탐색


```python
print('WindGustDir contains', len(df['WindGustDir'].u
```


```python
df['WindGustDir'].unique()
```


```python
df.WindGustDir.value_counts()
```


```python
pd.get_dummies(df.WindGustDir, drop_first=True, dumm
```


```python
pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).sum(axis=0)
```

WindDir9am 변수 탐색


```python
print('WindDir9am contains', len(df['WindDir9am'].
```


```python
df['WindDir9am'].unique()
```


```python
df['WindDir9am'].value_counts()
```


```python
pd.get_dummies(df.WindDir9am, drop_first=True, dumm
```


```python
pd.get_dummies(df.WindDir9am, drop_first=True, dumm
```

WindDir9am var에는 10013개의 결측값이 있음을 알 수 있습니다

WindDir3pm 변수 탐색


```python
print('WindDir3pm contains', len(df['WindDir3pm']
```


```python
df['WindDir3pm'].unique()
```


```python
df['WindDir3pm'].value_counts()
```


```python
pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_
```


```python
pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_
```

WindDir3pm 변수에는 3778개의 결측값이 있다.

RainToday 변수 탐색


```python
print('RainToday contains', len(df['RainToday'].uni
```


```python
df['RainToday'].unique()
```


```python
df.RainToday.value_counts()
```


```python
pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).head()
```


```python
pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).sum(axis=0)
```

RainToday 변수에는 1406개의 결측값이 있다.

수치형 변수 탐색


```python
numerical = [var for var in df.columns if df[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)
```


```python
df[numerical].head()
```

수치 변수 내의 문제 탐색

숫자 변수의 결측값


```python
df[numerical].isnull().sum()     #16개의 수치 변수에 결측값이 모두 포함
```

숫자 변수의 특이치


```python
print(round(df[numerical].describe()),2)
```

자세히 살펴보면 강우량, 증발량, 풍속 9am 및 풍속 3pm 열에 특이치가 포함되어 있을 수 있습니다.

상자 그림을 그려 위 변수의 특이치를 시각화


```python
plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')


plt.subplot(2, 2, 2)
fig = df.boxplot(column='Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')


plt.subplot(2, 2, 3)
fig = df.boxplot(column='WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')


plt.subplot(2, 2, 4)
fig = df.boxplot(column='WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')
```

 ![nn](data/tnclgud.png)

변수 분포 확인
이제 히스토그램을 그려 분포가 정규 분포인지 치우쳐 있는지 확인합니다. 변수가 정규 분포를 따르는 경우 극단값 분석을 수행하고, 그렇지 않은 경우 치우친 경우 IQR(양자 간 범위)을 찾습니다.


```python
plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.Rainfall.hist(bins=10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 2)
fig = df.Evaporation.hist(bins=10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 3)
fig = df.WindSpeed9am.hist(bins=10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 4)
fig = df.WindSpeed3pm.hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('RainTomorrow')
```

 ![nn](./333.png)

네 가지 변수가 모두 치우쳐 있음을 알 수 있습니다. 따라서 특이치를 찾기 위해 분위수 범위를 사용할 것


```python
IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)
Lower_fence = df.Rainfall.quantile(0.25) - (IQR * 3)
Upper_fence = df.Rainfall.quantile(0.75) + (IQR * 3)
print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```


```python
강우량의 경우 최소값과 최대값은 0.0과 371.0입니다. 따라서 특이치는 3.2보다 큰 값입니다.
```


```python
IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)
Lower_fence = df.Evaporation.quantile(0.25) - (IQR * 3)
Upper_fence = df.Evaporation.quantile(0.75) + (IQR * 3)
print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

증발의 경우 최소값과 최대값은 0.0과 145.0입니다. 따라서 특이치는 21.8보다 큰 값입니다.


```python
IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)
Lower_fence = df.WindSpeed9am.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed9am.quantile(0.75) + (IQR * 3)
print('WindSpeed9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

풍속 9am의 경우 최소값과 최대값은 0.0과 130.0입니다. 따라서 특이치는 55.0보다 큰 값입니다.


```python
IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)
Lower_fence = df.WindSpeed3pm.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed3pm.quantile(0.75) + (IQR * 3)
print('WindSpeed3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

풍속 3pm의 경우 최소값과 최대값은 0.0과 87.0입니다. 따라서 특이치는 57.0보다 큰 값입니다.

8. 피쳐 벡터 및 대상 변수 선언


```python
X = df.drop(['RainTomorrow'], axis=1)

y = df['RainTomorrow']
```

9. 데이터를 별도의 교육 및 테스트 세트로 분할


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```


```python
X_train.shape, X_test.shape
```

10. 피쳐 엔지니어링

기능 엔지니어링은 원시 데이터를 유용한 기능으로 변환하여 모델을 더 잘 이해하고 예측력을 높이는 데 도움이 됩니다. 저는 다양한 유형의 변수에 대해 피쳐 엔지니어링을 수행할 것입니다.


```python
X_train.dtypes
```


```python
categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical
```


```python
numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical
```

숫자 변수의 결측값 엔지니어링


```python
X_train[numerical].isnull().sum()
```


```python
X_test[numerical].isnull().sum()
```


```python
for col in numerical:
    if X_train[col].isnull().mean()>0:
        print(col, round(X_train[col].isnull().mean(),4))
```

추정
데이터가 랜덤(MCAR)에서 완전히 누락되었다고 가정합니다.누락된 값을 가져올 수 있는 두 가지 방법이 있습니다.하나는 평균 또는 중위수분열을 의미하거나 다른 하나는 랜덤 표본 추출입니다.데이터 세트에 특이치가 있는 경우 중위수를 사용해야 합니다.따라서 중위수가 특이치에게 견고하기 때문에 중위수를 사용할 것입니다.

이 경우 이 경우 데이터의 적절한 통계 측정값을 가져올 것입니다.훈련 세트에 대한 선은 훈련 세트와 테스트 세트에 전파되어야 합니다.기차와 시험 세트 모두에서 누락된 값을 채우기 위해 통계적 조치가 사용되어야 한다고 말한다.이건 피할 수 없는 일입니다.


```python
for df1 in [X_train, X_test]:
    for col in numerical:
        col_median=X_train[col].median()
        df1[col].fillna(col_median, inplace=True)   
```


```python
X_train[numerical].isnull().sum()
```


```python
X_test[numerical].isnull().sum()
```

추정
데이터가 랜덤(MCAR)에서 완전히 누락되었다고 가정합니다.누락된 값을 가져올 수 있는 두 가지 방법이 있습니다.하나는 평균 또는 중위수분열을 의미하거나 다른 하나는 랜덤 표본 추출입니다.데이터 세트에 특이치가 있는 경우 중위수를 사용해야 합니다.따라서 중위수가 특이치에게 견고하기 때문에 중위수를 사용할 것입니다.

이 경우 이 경우 데이터의 적절한 통계 측정값을 가져올 것입니다.훈련 세트에 대한 선은 훈련 세트와 테스트 세트에 전파되어야 합니다.기차와 시험 세트 모두에서 누락된 값을 채우기 위해 통계적 조치가 사용되어야 한다고 말한다.이건 피할 수 없는 일입니다.


```python
for df1 in [X_train, X_test]:
    for col in numerical:
        col_median=X_train[col].median()
        df1[col].fillna(col_median, inplace=True)          
```


```python
X_train[numerical].isnull().sum()
```


```python
X_test[numerical].isnull().sum()
```

범주형 변수의 결측값 엔지니어링


```python
X_train[categorical].isnull().mean()
```


```python
for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, (X_train[col].isnull().mean()))
```


```python
for df2 in [X_train, X_test]:
    df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)
    df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
    df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)
    df2['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=T
```


```python
X_train[categorical].isnull().sum()
```


```python
X_test[categorical].isnull().sum()
```

X_train과 X_test에서 결측값을 확인


```python
X_train.isnull().sum()
```


```python
X_test.isnull().sum()
```

숫자 변수의 공학적 특이치
강우량, 증발량, 풍속 9am 및 풍속 3pm 열에 특이치가 포함되어 있는 것을 확인했습니다. 최상위 코드화 방법을 사용하여 최대값을 상한으로 설정하고 위 변수에서 특이치를 제거합니다.


```python
def max_value(df3, variable, top):
    return np.where(df3[variable]>top, top, df3[variable])

for df3 in [X_train, X_test]:
    df3['Rainfall'] = max_value(df3, 'Rainfall', 3.2)
    df3['Evaporation'] = max_value(df3, 'Evaporation', 21.8)
    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)
    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3
```


```python
X_train.Rainfall.max(), X_test.Rainfall.max()
```


```python
X_train.Evaporation.max(), X_test.Evaporation.max()
```


```python
X_train.WindSpeed9am.max(), X_test.WindSpeed9am.max()
```


```python
X_train.WindSpeed3pm.max(), X_test.WindSpeed3pm.max()
```


```python
X_train[numerical].describe()
```

범주형 변수 인코딩


```python
categorical
```


```python
X_train[categorical].head()
```


```python
import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['RainToday'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)
```


```python
X_train.head()
```

RainToday_0 및 RainToday_1 변수 두 개가 RainToday 변수에서 생성되었음을 알 수 있습니다.

X_train 훈련셋을 생성


```python
X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_train.Location), 
                     pd.get_dummies(X_train.WindGustDir),
                     pd.get_dummies(X_train.WindDir9am),
                     pd.get_dummies(X_train.WindDir3pm)], axis=1)
```


```python
X_train.head()
```

X_test testing set 생성


```python
X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_test.Location), 
                     pd.get_dummies(X_test.WindGustDir),
                     pd.get_dummies(X_test.WindDir9am),
                     pd.get_dummies(X_test.WindDir3pm)], axis=1)
```


```python
X_test.head()
```

이제 모델 구축을 위한 교육 및 테스트가 준비되었습니다. 그 전에 모든 형상 변수를 동일한 척도에 매핑해야 합니다. 이를 형상 스케일링이라고 합니다.

11. 피쳐 스케일링


```python
X_train.describe()
```


```python
cols = X_train.columns
```


```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
```


```python
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])
```


```python
X_train.describe()
```


```python
이제 X_train 데이터 세트를 로지스틱 회귀 분류기에 입력 가능
```

12. 모델 훈련


```python
# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression


# instantiate the model
logreg = LogisticRegression(solver='liblinear', random_state=0)


# fit the model
logreg.fit(X_train, y_train)
```

13. 결과 예측


```python
y_pred_test = logreg.predict(X_test)

y_pred_test
```

predict_proba 방법
predict_proba 메서드는 이 경우 대상 변수(0 및 1)에 대한 확률을 배열 형식으로 제공합니다.

0은 비가 오지 않을 확률이고 1은 비가 올 확률입니다.


```python
logreg.predict_proba(X_test)[:,0]
```


```python
logreg.predict_proba(X_test)[:,1]
```

14.정확도 점수 확인


```python
from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))
```

여기서 y_test는 참 클래스 레이블이고 y_pred_test는 테스트 세트의 예측 클래스 레이블입니다.

열차 세트와 테스트 세트 정확도 비교


```python
y_pred_train = logreg.predict(X_train)

y_pred_train
```


```python
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
```

과적합 및 과소적합 여부 점검


```python
print('Training set score: {:.4f}'.format(logreg.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg.score(X_test, y_test)))
```

교육 세트 정확도 점수는 0.8476인 반면 테스트 세트 정확도는 0.8501입니다. 이 두 값은 상당히 비슷합니다. 따라서 과적합의 문제는 없습니다.

로지스틱 회귀 분석에서는 C = 1의 기본값을 사용합니다. 교육 및 테스트 세트 모두에서 약 85%의 정확도로 우수한 성능을 제공합니다. 그러나 교육 및 테스트 세트의 모델 성능은 매우 유사합니다. 그것은 아마도 부족한 경우일 것입니다. 여기서는 C를 좀 더 늘리기로 했습니다.


```python
logreg100 = LogisticRegression(C=100, solver='liblinear', random_state=0)


# fit the model
logreg100.fit(X_train, y_train)
```


```python
print('Training set score: {:.4f}'.format(logreg100.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg100.score(X_test, y_test)))
```

우리는 C=100이 테스트 세트 정확도를 높이고 교육 세트 정확도를 약간 높인다는 것을 알 수 있습니다. 따라서 더 복잡한 모델이 더 나은 성능을 발휘해야 한다는 결론을 내릴 수 있습니다.

이제 C=0.01을 설정하여 기본값인 C=1보다 정규화된 모델을 사용하면 어떻게 되는지 알아보겠습니다.


```python
logreg001 = LogisticRegression(C=0.01, solver='liblinear', random_state=0)


# fit the model
logreg001.fit(X_train, y_train)
```


```python
print('Training set score: {:.4f}'.format(logreg001.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg001.score(X_test, y_test)))
```

따라서 C=0.01을 설정하여 보다 정규화된 모델을 사용하면 교육 및 테스트 세트 정확도가 기본 매개 변수에 비해 모두 감소합니다.

모델 정확도와 null 정확도 비교
따라서 모형 정확도는 0.8501입니다. 그러나 위의 정확도에 근거하여 우리의 모델이 매우 좋다고 말할 수는 없습니다. 우리는 그것을 null 정확도와 비교해야 합니다. Null 정확도는 항상 가장 빈도가 높은 클래스를 예측하여 얻을 수 있는 정확도입니다.

그래서 우리는 먼저 테스트 세트의 클래스 분포를 확인해야 합니다.


```python
y_test.value_counts()
```

우리는 가장 빈번한 수업의 발생 횟수가 22067회임을 알 수 있습니다. 따라서 22067을 총 발생 횟수로 나누어 null 정확도를 계산할 수 있습니다.


```python
null_accuracy = (22067/(22067+6372))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))
```

우리의 모델 정확도 점수는 0.8501이지만 null 정확도 점수는 0.7759임을 알 수 있습니다. 따라서 로지스틱 회귀 분석 모형이 클래스 레이블을 예측하는 데 매우 효과적이라는 결론을 내릴 수 있습니다.

이제 위의 분석을 바탕으로 분류 모델 정확도가 매우 우수하다는 결론을 내릴 수 있습니다. 우리 모델은 클래스 레이블을 예측하는 측면에서 매우 잘 수행하고 있습니다.

그러나 기본적인 값 분포는 제공하지 않습니다. 또한, 그것은 우리 반 학생들이 저지르는 오류의 유형에 대해서는 아무 것도 말해주지 않습니다. 우리는 Confusion matrix라 불리는 또 다른 도구가 있습니다.

15. 오차 행렬

오차 행렬은 분류 알고리즘의 성능을 요약하는 도구입니다. 혼동 행렬은 분류 모델 성능과 모델에 의해 생성되는 오류 유형에 대한 명확한 그림을 제공합니다. 각 범주별로 분류된 정확한 예측과 잘못된 예측의 요약을 제공합니다. 요약은 표 형식으로 표시됩니다.

분류 모델 성능을 평가하는 동안 네 가지 유형의 결과가 가능합니다. 이 네 가지 결과는 아래에 설명되어 있습니다

참 양성(TP) – 참 양성은 관측치가 특정 클래스에 속하고 관측치가 실제로 해당 클래스에 속한다고 예측할 때 발생합니다.

True Negatives(TN) – True Negatives는 관측치가 특정 클래스에 속하지 않고 실제로 관측치가 해당 클래스에 속하지 않을 때 발생합니다.

False Positives(FP) – False Positives는 관측치가 특정 클래스에 속하지만 실제로는 해당 클래스에 속하지 않는다고 예측할 때 발생합니다. 이러한 유형의 오류를 유형 I 오류라고 합니다.

FN(False Negatives) – 관측치가 특정 클래스에 속하지 않지만 실제로는 해당 클래스에 속한다고 예측할 때 False Negatives가 발생합니다. 이것은 매우 심각한 오류이며 Type II 오류라고 합니다.


```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_test)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])
```

혼동 행렬은 20892 + 3285 = 24177 정확한 예측과 3087 + 1175 = 4262 부정확한 예측을 나타냅니다.

참 양성(실제 양성:1 및 예측 양성:1) - 20892
참 음수(실제 음수:0 및 예측 음수:0) - 3285
거짓 양성(실제 음성: 0이지만 예측 양성: 1) - 1175(유형 I 오류)
거짓 음성(실제 양의 1이지만 예측 음의 0) - 3087(타입 II 오류)


```python
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
```


```python
16.분류측정지표
```

분류 보고서
분류 보고서는 분류 모델 성능을 평가하는 또 다른 방법입니다. 모형의 정밀도, 호출, f1 및 지원 점수가 표시됩니다.


```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_test))
```

분류정확도


```python
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
```


```python
classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
```

분류오류


```python
classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))
```

정밀도
정밀도는 모든 예측 긍정 결과 중 정확하게 예측된 긍정 결과의 백분율로 정의할 수 있습니다. 참 및 거짓 양성의 합계에 대한 참 양성(TP + FP)의 비율로 지정할 수 있습니다.

따라서 정밀도는 정확하게 예측된 양성 결과의 비율을 식별합니다. 그것은 부정적인 계층보다 긍정적인 계층에 더 관심이 있습니다.

수학적으로 정밀도는 TP/(TP + FP)의 비율로 정의할 수 있습니다.


```python
precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))
```


```python
재현율
리콜은 모든 실제 긍정적 결과 중 정확하게 예측된 긍정적 결과의 비율로 정의할 수 있습니다. 참 양성과 거짓 음성의 합(TP + FN)에 대한 참 양성(TP)의 비율로 지정할 수 있습니다. 리콜은 민감도라고도 합니다.

호출은 정확하게 예측된 실제 긍정의 비율을 식별합니다.

수학적으로 호출은 TP/(TP + FN)의 비율로 지정할 수 있습니다.
```


```python
recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))
```


```python
#True Positive Rate(=재현율)
true_positive_rate = TP / float(TP + FN)


print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
```


```python
#False Positive Rate
false_positive_rate = FP / float(FP + TN)


print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
```


```python
#Specificity
specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))
```

f1 score
f1-점수는 정밀도와 호출의 가중 조화 평균입니다. 가능한 최상의 f1-점수는 1.0이고 최악의 f1-점수는 0.0입니다. f1-점수는 정밀도와 호출의 조화 평균입니다. 따라서 f1-점수는 정확도와 리콜을 계산에 포함시키기 때문에 정확도 측도보다 항상 낮습니다. f1-점수의 가중 평균은 전역 정확도가 아닌 분류기 모델을 비교하는 데 사용되어야 합니다.

Support
데이터 세트에서 클래스의 실제 발생 횟수입니다.

17. 임계값 레벨 조정


```python
y_pred_prob = logreg.predict_proba(X_test)[0:10]

y_pred_prob
```

관찰
- 각 행에서 숫자는 1이 됩니다.
- 2개의 클래스(0 및 1)에 해당하는 2개의 열이 있습니다.

    -클래스 0 - 내일 비가 오지 않을 확률을 예측합니다.
    -클래스 1 - 내일 비가 올 확률을 예측합니다.

- 예측 확률의 중요성
비가 오거나 오지 않을 확률로 관측치의 순위를 매길 수 있습니다.

- predict_proba 공정
    - 확률을 예측
    - 확률이 가장 높은 클래스 선택

- 분류 임계값 레벨
    -분류 임계값 레벨은 0.5입니다.
    -클래스 1 - 확률이 0.5 이상일 경우 비가 올 확률이 예측됩니다.
    -클래스 0 - 확률이 0.5 미만일 경우 비가 오지 않을 확률이 예측됩니다.


```python
y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - No rain tomorrow (0)', 'Prob of - Rain tomorrow (1)'])

y_pred_prob_df
```


```python
logreg.predict_proba(X_test)[0:10, 1]
```


```python
y_pred1 = logreg.predict_proba(X_test)[:, 1]
```


```python
plt.rcParams['font.size'] = 12


# plot histogram with 10 bins
plt.hist(y_pred1, bins = 10)


# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of rain')


# set the x-axis limit
plt.xlim(0,1)


# set the title
plt.xlabel('Predicted probabilities of rain')
plt.ylabel('Frequency')
```

관찰
- 위의 히스토그램이 매우 양으로 치우쳐 있음을 알 수 있습니다.
- 첫 번째 열은 확률이 0.0과 0.1 사이인 관측치가 약 15,000개임을 나타냅니다.
- 확률이 0.5보다 작은 관측치가 있습니다.
- 그래서 이 소수의 관측치들은 내일 비가 올 것이라고 예측하고 있습니다.
- 내일은 비가 오지 않을 것이라는 관측이 대다수입니다.

임계값 인하


```python
from sklearn.preprocessing import binarize

for i in range(1,5):
    
    cm1=0
    
    y_pred1 = logreg.predict_proba(X_test)[:,1]
    
    y_pred1 = y_pred1.reshape(-1,1)
    
    y_pred2 = binarize(y_pred1, i/10)
    
    y_pred2 = np.where(y_pred2 == 1, 'Yes', 'No')
    
    cm1 = confusion_matrix(y_test, y_pred2)
        
    print ('With',i/10,'threshold the Confusion Matrix is ','\n\n',cm1,'\n\n',
           
            'with',cm1[0,0]+cm1[1,1],'correct predictions, ', '\n\n', 
           
            cm1[0,1],'Type I errors( False Positives), ','\n\n',
           
            cm1[1,0],'Type II errors( False Negatives), ','\n\n',
           
           'Accuracy score: ', (accuracy_score(y_test, y_pred2)), '\n\n',
           
           'Sensitivity: ',cm1[1,1]/(float(cm1[1,1]+cm1[1,0])), '\n\n',
           
           'Specificity: ',cm1[0,0]/(float(cm1[0,0]+cm1[0,1])),'\n\n',
          
            '====================================================', '\n\n')
```

18. ROC - AUC

ROC 곡선
분류 모델 성능을 시각적으로 측정하는 또 다른 도구는 ROC 곡선입니다. ROC 곡선은 수신기 작동 특성 곡선을 나타냅니다. ROC 곡선은 다양한 분류 임계값 수준에서 분류 모델의 성능을 보여주는 그림입니다.

ROC 곡선은 다양한 임계값 레벨에서 FPR(False Positive Rate)에 대한 True Positive Rate(TPR)를 표시합니다.

실제 양의 비율(TPR)은 재현율이라고도 합니다. TP 대 (TP + FN)의 비율로 정의됩니다.
FPR(False Positive Rate)은 FP 대 (FP + TN)의 비율로 정의됩니다.

ROC 곡선에서는 단일 지점의 TPR(True Positive Rate)과 FPR(False Positive Rate)에 초점을 맞출 것입니다. 이를 통해 다양한 임계값 레벨에서 TPR과 FPR로 구성된 ROC 곡선의 일반적인 성능을 얻을 수 있습니다. 따라서 ROC 곡선은 여러 분류 임계값 수준에서 TPR 대 FPR을 표시합니다. 임계값 레벨을 낮추면 더 많은 항목이 포지티브로 분류될 수 있습니다. 그러면 True Positives(TP)와 False Positives(FP)가 모두 증가합니다.


```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred1, pos_label = 'Yes')

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for RainTomorrow classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()
```

ROC 곡선은 특정 컨텍스트에 대한 민감도와 특수성의 균형을 맞추는 임계값 레벨을 선택하는 데 도움

ROC-AUC
ROC AUC는 수신기 작동 특성 - 곡선 아래 영역을 나타냅니다. 분류기 성능을 비교하는 기술입니다. 이 기술에서 우리는 곡선 아래의 면적을 측정합니다. 완벽한 분류기는 ROC AUC가 1인 반면, 순수한 무작위 분류기는 ROC AUC가 0.5입니다.

즉, ROC AUC는 곡선 아래에 있는 ROC 그림의 백분율입니다.


```python
from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred1)

print('ROC AUC : {:.4f}'.format(ROC_AUC))
```


```python
from sklearn.model_selection import cross_val_score

Cross_validated_ROC_AUC = cross_val_score(logreg, X_train, y_train, cv=5, scoring='roc_auc').mean()

print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))
```


```python
19. k-Fold 교차 검증
```


```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(logreg, X_train, y_train, cv = 5, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))
```

평균을 계산하여 교차 검증 정확도를 요약할 수 있습니다.


```python
print('Average cross-validation score: {:.4f}'.format(scores.mean()))
```

우리의 원래 모델 점수는 0.8476입니다. 교차 검증 평균 점수는 0.8474입니다. 따라서 교차 검증을 통해 성능이 향상되지 않는다는 결론을 내릴 수 있습니다.

20. GridSearch CV를 이용한 하이퍼파라미터 최적화


```python
from sklearn.model_selection import GridSearchCV


parameters = [{'penalty':['l1','l2']}, 
              {'C':[1, 10, 100, 1000]}]



grid_search = GridSearchCV(estimator = logreg,  
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           verbose=0)


grid_search.fit(X_train, y_train)
```


```python
print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))

# print parameters that give the best results
print('Parameters that give the best results :','\n\n', (grid_search.best_params_))

# print estimator that was chosen by the GridSearch
print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))
```


```python
print('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_test, y_test)))
```

21. 결과 및 결론

 로지스틱 회귀 모형 정확도 점수는 0.8501입니다. 그래서, 이 모델은 호주에 내일 비가 올지 안 올지 예측하는 데 매우 좋은 역할을 합니다.

 내일 비가 올 것이라는 관측은 소수입니다. 내일은 비가 오지 않을 것이라는 관측이 대다수입니다.

 모형에 과적합 징후가 없습니다.

 C 값을 늘리면 테스트 세트 정확도가 높아지고 교육 세트 정확도가 약간 높아집니다. 따라서 더 복잡한 모델이 더 나은 성능을 발휘해야 한다는 결론을 내릴 수 있습니다.

임계값 레벨을 높이면 정확도가 높아집니다.

우리 모델의 ROCAUC는 1에 접근합니다. 그래서, 우리는 우리의 분류기가 내일 비가 올지 안 올지 예측하는 것을 잘한다는 결론을 내릴 수 있습니다.

우리의 원래 모델 정확도 점수는 0.8501인 반면 RFECV 이후 정확도 점수는 0.8500입니다. 따라서 기능 집합을 줄이면 거의 유사한 정확도를 얻을 수 있습니다.

원래 모델에서는 FP = 1175인 반면 FP1 = 1174입니다. 그래서 우리는 대략 같은 수의 오검출을 얻습니다. 또한 FN = 3087인 반면 FN1 = 3091입니다. 그래서 우리는 약간 더 높은 거짓 음성을 얻습니다.

우리의 원래 모델 점수는 0.8476입니다. 교차 검증 평균 점수는 0.8474입니다. 따라서 교차 검증을 통해 성능이 향상되지 않는다는 결론을 내릴 수 있습니다.

우리의 원래 모델 테스트 정확도는 0.8501인 반면 그리드 검색 CV 정확도는 0.8507입니다. 그리드 검색 CV가 이 특정 모델의 성능을 향상시킨다는 것을 알 수 있습니다.
