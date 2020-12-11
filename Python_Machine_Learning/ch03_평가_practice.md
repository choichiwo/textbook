## 06. 피마 인디언 당뇨병 예측

피마 인디언 당뇨병 데이터 세트 구성 살펴보기  
- Pregnancies: 임신횟수  
- Glucose: 포도당 부하 검사 수치  
- BloodPressure: 혈압  
- SkinThickness: 팔 삼두근 뒤쪽의 피하지방 측정값  
- Insulin: 혈청 인슐린  
- BMI: 체질량 지수  
- DiabetesPedigreeFunction : 당뇨 내력 가중치 값  
- Age: 나이   
- Outcome: 당뇨여부(0 또는 1)  


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler, Binarizer
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')
```


```python
# 데이터 불러오기
diabetes_data = pd.read_csv('../data/pima-indians/diabetes.csv')
print(diabetes_data['Outcome'].value_counts())
diabetes_data.head(3)
```

    0    500
    1    268
    Name: Outcome, dtype: int64
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



- 전체 768개의 데이터 중, Negative 값 0이 500개, Positive 값 1이 268개


```python
# feature 타입과 Null 개수 세어보기
diabetes_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 768 entries, 0 to 767
    Data columns (total 9 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   Pregnancies               768 non-null    int64  
     1   Glucose                   768 non-null    int64  
     2   BloodPressure             768 non-null    int64  
     3   SkinThickness             768 non-null    int64  
     4   Insulin                   768 non-null    int64  
     5   BMI                       768 non-null    float64
     6   DiabetesPedigreeFunction  768 non-null    float64
     7   Age                       768 non-null    int64  
     8   Outcome                   768 non-null    int64  
    dtypes: float64(2), int64(7)
    memory usage: 54.1 KB
    

- Null 값은 없으며 피처 타입은 모두 숫자형  

**-** 로지스틱 회귀를 이용해 예측 모델 생성하기


```python
# 평가지표 출력하는 함수 설정
def get_clf_eval(y_test, y_pred):
    confusion = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    
    print('오차행렬:\n', confusion)
    print('\n정확도: {:.4f}'.format(accuracy))
    print('정밀도: {:.4f}'.format(precision))
    print('재현율: {:.4f}'.format(recall))
    print('F1: {:.4f}'.format(F1))
    print('AUC: {:.4f}'.format(AUC))
```


```python
# Precision-Recall Curve Plot 그리기
def precision_recall_curve_plot(y_test, pred_proba):
    # threshold ndarray와 이 threshold에 따른 정밀도, 재현율 ndarray 추출
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba)
    
    # x축을 threshold, y축을 정밀도, 재현율로 그래프 그리기
    plt.figure(figsize=(8, 6))
    thresholds_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[:thresholds_boundary], linestyle='--', label='precision')
    plt.plot(thresholds, recalls[:thresholds_boundary], linestyle=':', label='recall')
    
    # threshold의 값 X축의 scale을 0.1 단위로 변경
    stard, end = plt.xlim()
    plt.xticks(np.round(np.arange(stard, end, 0.1), 2))
    
    plt.xlim()
    plt.xlabel('thresholds')
    plt.ylabel('precision & recall value')
    plt.legend()
    plt.grid()
```


```python
# 피처 데이터 세트 X, 레이블 데이터 세트 y 추출
# 맨 끝이 Outcome 칼럼으로 레이블 값, 칼럼 위치 -1을 이용해 추출
X = diabetes_data.iloc[:, :-1]
y = diabetes_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 156, stratify=y)

# 로지스틱 회귀로 학습, 예측 및 평가 수행
lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
pred = lr_clf.predict(X_test)
get_clf_eval(y_test, pred)
```

    오차행렬:
     [[88 12]
     [23 31]]
    
    정확도: 0.7727
    정밀도: 0.7209
    재현율: 0.5741
    F1: 0.6392
    AUC: 0.7270
    


```python
# 임계값별로 정밀도 - 재현율 출력
pred_proba = lr_clf.predict_proba(X_test)[:, 1]
precision_recall_curve_plot(y_test, pred_proba)
```


![png](output_9_0.png)


- 재현율 곡선을 보면 임계값을 0.42 정도로 낮추면 정밀도와 재현율이 어느 정도 균형을 맞출 것  
  그러나, 두 지표 모두 0.7이 되지 않는 수치  


**-** 임계값을 인위 조작하기 전에 다시 데이터 점검해보기


```python
# 원본 데이터 DataFrane describe() 메서드로 피처 값의 분포도 살피기
diabetes_data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.845052</td>
      <td>120.894531</td>
      <td>69.105469</td>
      <td>20.536458</td>
      <td>79.799479</td>
      <td>31.992578</td>
      <td>0.471876</td>
      <td>33.240885</td>
      <td>0.348958</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.369578</td>
      <td>31.972618</td>
      <td>19.355807</td>
      <td>15.952218</td>
      <td>115.244002</td>
      <td>7.884160</td>
      <td>0.331329</td>
      <td>11.760232</td>
      <td>0.476951</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.078000</td>
      <td>21.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>99.000000</td>
      <td>62.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>27.300000</td>
      <td>0.243750</td>
      <td>24.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>117.000000</td>
      <td>72.000000</td>
      <td>23.000000</td>
      <td>30.500000</td>
      <td>32.000000</td>
      <td>0.372500</td>
      <td>29.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.000000</td>
      <td>140.250000</td>
      <td>80.000000</td>
      <td>32.000000</td>
      <td>127.250000</td>
      <td>36.600000</td>
      <td>0.626250</td>
      <td>41.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>17.000000</td>
      <td>199.000000</td>
      <td>122.000000</td>
      <td>99.000000</td>
      <td>846.000000</td>
      <td>67.100000</td>
      <td>2.420000</td>
      <td>81.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


min() 값이 0으로 된 피처가 상당히 많음  
Glucose 피처는 포도당 수치로 min 값이 0으로 나올 수는 없음

```python
# min() 값이 0으로 된 피처에 대해 0 값의 건수 및 전체 데이터 건수 대비 몇 퍼센트의 비율로 존재하는지 확인해보기
feature_list = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

def hist_plot(df):
    for col in feature_list:
        df[col].plot(kind='hist', bins=20).set_title('Histogram of '+col)
        plt.show()

hist_plot(diabetes_data)
```


![png](output_13_0.png)



![png](output_13_1.png)



![png](output_13_2.png)



![png](output_13_3.png)



![png](output_13_4.png)


- SkinThickness와 Insulin의 0 값은 전체의 29.56%, 48.7%로 많은 수준  

**-** 일괄 삭제 대신 위 피처의 0 값을 평균값으로 대체


```python
# 위 컬럼들에 대한 0 값의 비율 확인
zero_count = []
zero_percent = []
for col in feature_list:
    zero_num = diabetes_data[diabetes_data[col]==0].shape[0]
    zero_count.append(zero_num)
    zero_percent.append(np.round(zero_num/diabetes_data.shape[0]*100,2))

zero = pd.DataFrame([zero_count, zero_percent], columns=feature_list, index=['count', 'percent']).T
zero
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Glucose</th>
      <td>5.0</td>
      <td>0.65</td>
    </tr>
    <tr>
      <th>BloodPressure</th>
      <td>35.0</td>
      <td>4.56</td>
    </tr>
    <tr>
      <th>SkinThickness</th>
      <td>227.0</td>
      <td>29.56</td>
    </tr>
    <tr>
      <th>Insulin</th>
      <td>374.0</td>
      <td>48.70</td>
    </tr>
    <tr>
      <th>BMI</th>
      <td>11.0</td>
      <td>1.43</td>
    </tr>
  </tbody>
</table>
</div>




```python
# zero_features 리스트 내부 저장된 개별 피처의 0 값을 NaN 값으로 대체
diabetes_data[feature_list] = diabetes_data[feature_list].replace(0, np.nan)

# 위 5개 feature 에 대해 0값을 평균 값으로 대체
mean_features = diabetes_data[feature_list].mean()
diabetes_data[feature_list] = diabetes_data[feature_list].replace(np.nan, mean_features)
```


```python
# 데이터 세트에 피처 스케일링을 적용해 변환하기
# 로지스틱 회귀의 경우, 숫자 데이터에 스케일링을 적용하는 것이 일반적으로 성능이 좋음
X = diabetes_data.iloc[:, :-1]
y = diabetes_data.iloc[:, -1]

# StandardScaler 클래스를 상용하여 데이터 세트에 스케일링 적용
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state=156, stratify = y)

# 로지스틱 회귀로 학습, 예측, 평가 수행
lr_clf = LogisticRegression()
lr_clf.fit(X_train,  y_train)
pred = lr_clf.predict(X_test)
get_clf_eval(y_test, pred)
```

    오차행렬:
     [[89 11]
     [21 33]]
    
    정확도: 0.7922
    정밀도: 0.7500
    재현율: 0.6111
    F1: 0.6735
    AUC: 0.7506
    


```python
# 평가지표를 조사하기 위한 새로운 함수 생성
def get_eval_by_threshold(y_test, pred_proba_c1, thresholds):
    #thresholds list 객체 내의 값을 iteration 하면서 평가 수행
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1)
        custom_predict = binarizer.transform(pred_proba_c1)
        print('\n임계값: ', custom_threshold)
        get_clf_eval(y_test, custom_predict)
```

- 데이터 변환과 스케일링으로 성능 수치가 일정 수준 개선  
  하지만, 여전히 재현율 수치 개선이 필요함
  
**-** 분류 결정 임계값을 변화시키면서 재현율 값의 성능 수치 개선 정도를 확인하기


```python
# 임계값을 0.3에서 0.5까지 0.03씩 변화시키면서 재현율과 다른 평가 지표의 값 변화를 출력
thresholds = [0.3, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.50]
pred_proba = lr_clf.predict_proba(X_test)
get_eval_by_threshold(y_test, pred_proba[:, 1].reshape(-1, 1), thresholds)
```

    
    임계값:  0.3
    오차행렬:
     [[68 32]
     [10 44]]
    
    정확도: 0.7273
    정밀도: 0.5789
    재현율: 0.8148
    F1: 0.6769
    AUC: 0.7474
    
    임계값:  0.33
    오차행렬:
     [[74 26]
     [11 43]]
    
    정확도: 0.7597
    정밀도: 0.6232
    재현율: 0.7963
    F1: 0.6992
    AUC: 0.7681
    
    임계값:  0.36
    오차행렬:
     [[75 25]
     [13 41]]
    
    정확도: 0.7532
    정밀도: 0.6212
    재현율: 0.7593
    F1: 0.6833
    AUC: 0.7546
    
    임계값:  0.39
    오차행렬:
     [[82 18]
     [16 38]]
    
    정확도: 0.7792
    정밀도: 0.6786
    재현율: 0.7037
    F1: 0.6909
    AUC: 0.7619
    
    임계값:  0.42
    오차행렬:
     [[85 15]
     [18 36]]
    
    정확도: 0.7857
    정밀도: 0.7059
    재현율: 0.6667
    F1: 0.6857
    AUC: 0.7583
    
    임계값:  0.45
    오차행렬:
     [[86 14]
     [19 35]]
    
    정확도: 0.7857
    정밀도: 0.7143
    재현율: 0.6481
    F1: 0.6796
    AUC: 0.7541
    
    임계값:  0.48
    오차행렬:
     [[88 12]
     [20 34]]
    
    정확도: 0.7922
    정밀도: 0.7391
    재현율: 0.6296
    F1: 0.6800
    AUC: 0.7548
    
    임계값:  0.5
    오차행렬:
     [[89 11]
     [21 33]]
    
    정확도: 0.7922
    정밀도: 0.7500
    재현율: 0.6111
    F1: 0.6735
    AUC: 0.7506
    

- 0.33: 정확도와 정밀도를 희생하고 재현율을 높이는 데 가장 좋은 임계값  
- 0.48: 전체적인 성능 평가 지표를 유지하며 재현율을 약간 향상시키는 좋은 임계값  


**-** 앞서 학습된 로지스틱 회귀 모델을 이용해 임계값을 0.48로 낮춘 상태에서 다시 예측하기


```python
# 임계값을 0.48로 설정한 Binarizer 생성
binarizer = Binarizer(threshold=0.48)

# 위에서 구한 predict_proba() 예측 확률 array에서 1에 해당하는 칼럼값을 Binarizer 변환하기
pred_th_048 = binarizer.fit_transform(pred_proba[:, 1].reshape(-1, 1))

get_clf_eval(y_test, pred_th_048)
```

    오차행렬:
     [[88 12]
     [20 34]]
    
    정확도: 0.7922
    정밀도: 0.7391
    재현율: 0.6296
    F1: 0.6800
    AUC: 0.7548
    
