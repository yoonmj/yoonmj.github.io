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
