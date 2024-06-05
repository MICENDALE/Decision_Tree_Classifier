# Breast Cancer Prediction using Decision Tree Classifier

A decision tree is a non-parametric, supervised machine learning algorithm that is implemented to classify or predict a value by learning certain decision rules. The are two types of decision trees used for classification and regression purposes. For this project, we will train a decision tree classifier. 

## Aim
To build a predictor model that classifies and labels breast tumors as **benign** or **malignant**

## Approach 
1. Dataset will be obtained from the **Scikit-learn** library in python
2. One third of the dataset will be allocated for testing 
3. The *DecisionTreeClassifer* model will choose the best attributes to split the data
4. Stopping criteria will be determined, if required. 

Importing dataset 

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
dataset = pd.DataFrame(data=data['data'], columns=data['feature_names'])
dataset
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
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst radius</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.30010</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>...</td>
      <td>25.380</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.16220</td>
      <td>0.66560</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.08690</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>...</td>
      <td>24.990</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.12380</td>
      <td>0.18660</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.19740</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>...</td>
      <td>23.570</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.14440</td>
      <td>0.42450</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.24140</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>0.09744</td>
      <td>...</td>
      <td>14.910</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.20980</td>
      <td>0.86630</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.19800</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>0.05883</td>
      <td>...</td>
      <td>22.540</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.13740</td>
      <td>0.20500</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>564</th>
      <td>21.56</td>
      <td>22.39</td>
      <td>142.00</td>
      <td>1479.0</td>
      <td>0.11100</td>
      <td>0.11590</td>
      <td>0.24390</td>
      <td>0.13890</td>
      <td>0.1726</td>
      <td>0.05623</td>
      <td>...</td>
      <td>25.450</td>
      <td>26.40</td>
      <td>166.10</td>
      <td>2027.0</td>
      <td>0.14100</td>
      <td>0.21130</td>
      <td>0.4107</td>
      <td>0.2216</td>
      <td>0.2060</td>
      <td>0.07115</td>
    </tr>
    <tr>
      <th>565</th>
      <td>20.13</td>
      <td>28.25</td>
      <td>131.20</td>
      <td>1261.0</td>
      <td>0.09780</td>
      <td>0.10340</td>
      <td>0.14400</td>
      <td>0.09791</td>
      <td>0.1752</td>
      <td>0.05533</td>
      <td>...</td>
      <td>23.690</td>
      <td>38.25</td>
      <td>155.00</td>
      <td>1731.0</td>
      <td>0.11660</td>
      <td>0.19220</td>
      <td>0.3215</td>
      <td>0.1628</td>
      <td>0.2572</td>
      <td>0.06637</td>
    </tr>
    <tr>
      <th>566</th>
      <td>16.60</td>
      <td>28.08</td>
      <td>108.30</td>
      <td>858.1</td>
      <td>0.08455</td>
      <td>0.10230</td>
      <td>0.09251</td>
      <td>0.05302</td>
      <td>0.1590</td>
      <td>0.05648</td>
      <td>...</td>
      <td>18.980</td>
      <td>34.12</td>
      <td>126.70</td>
      <td>1124.0</td>
      <td>0.11390</td>
      <td>0.30940</td>
      <td>0.3403</td>
      <td>0.1418</td>
      <td>0.2218</td>
      <td>0.07820</td>
    </tr>
    <tr>
      <th>567</th>
      <td>20.60</td>
      <td>29.33</td>
      <td>140.10</td>
      <td>1265.0</td>
      <td>0.11780</td>
      <td>0.27700</td>
      <td>0.35140</td>
      <td>0.15200</td>
      <td>0.2397</td>
      <td>0.07016</td>
      <td>...</td>
      <td>25.740</td>
      <td>39.42</td>
      <td>184.60</td>
      <td>1821.0</td>
      <td>0.16500</td>
      <td>0.86810</td>
      <td>0.9387</td>
      <td>0.2650</td>
      <td>0.4087</td>
      <td>0.12400</td>
    </tr>
    <tr>
      <th>568</th>
      <td>7.76</td>
      <td>24.54</td>
      <td>47.92</td>
      <td>181.0</td>
      <td>0.05263</td>
      <td>0.04362</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.1587</td>
      <td>0.05884</td>
      <td>...</td>
      <td>9.456</td>
      <td>30.37</td>
      <td>59.16</td>
      <td>268.6</td>
      <td>0.08996</td>
      <td>0.06444</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.2871</td>
      <td>0.07039</td>
    </tr>
  </tbody>
</table>
<p>569 rows × 30 columns</p>
</div>

The dataset has 569 observations and 30 attributes that will be used to classify a breast cancer into either benign or malignant category. 

Now, let's split the dataset into training and testing dataframes. 

```python
from sklearn.model_selection import train_test_split
X = dataset.copy()
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
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
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst radius</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>131</th>
      <td>15.46</td>
      <td>19.48</td>
      <td>101.70</td>
      <td>748.9</td>
      <td>0.10920</td>
      <td>0.12230</td>
      <td>0.14660</td>
      <td>0.08087</td>
      <td>0.1931</td>
      <td>0.05796</td>
      <td>...</td>
      <td>19.26</td>
      <td>26.00</td>
      <td>124.90</td>
      <td>1156.0</td>
      <td>0.1546</td>
      <td>0.2394</td>
      <td>0.37910</td>
      <td>0.15140</td>
      <td>0.2837</td>
      <td>0.08019</td>
    </tr>
    <tr>
      <th>62</th>
      <td>14.25</td>
      <td>22.15</td>
      <td>96.42</td>
      <td>645.7</td>
      <td>0.10490</td>
      <td>0.20080</td>
      <td>0.21350</td>
      <td>0.08653</td>
      <td>0.1949</td>
      <td>0.07292</td>
      <td>...</td>
      <td>17.67</td>
      <td>29.51</td>
      <td>119.10</td>
      <td>959.5</td>
      <td>0.1640</td>
      <td>0.6247</td>
      <td>0.69220</td>
      <td>0.17850</td>
      <td>0.2844</td>
      <td>0.11320</td>
    </tr>
    <tr>
      <th>543</th>
      <td>13.21</td>
      <td>28.06</td>
      <td>84.88</td>
      <td>538.4</td>
      <td>0.08671</td>
      <td>0.06877</td>
      <td>0.02987</td>
      <td>0.03275</td>
      <td>0.1628</td>
      <td>0.05781</td>
      <td>...</td>
      <td>14.37</td>
      <td>37.17</td>
      <td>92.48</td>
      <td>629.6</td>
      <td>0.1072</td>
      <td>0.1381</td>
      <td>0.10620</td>
      <td>0.07958</td>
      <td>0.2473</td>
      <td>0.06443</td>
    </tr>
    <tr>
      <th>312</th>
      <td>12.76</td>
      <td>13.37</td>
      <td>82.29</td>
      <td>504.1</td>
      <td>0.08794</td>
      <td>0.07948</td>
      <td>0.04052</td>
      <td>0.02548</td>
      <td>0.1601</td>
      <td>0.06140</td>
      <td>...</td>
      <td>14.19</td>
      <td>16.40</td>
      <td>92.04</td>
      <td>618.8</td>
      <td>0.1194</td>
      <td>0.2208</td>
      <td>0.17690</td>
      <td>0.08411</td>
      <td>0.2564</td>
      <td>0.08253</td>
    </tr>
    <tr>
      <th>377</th>
      <td>13.46</td>
      <td>28.21</td>
      <td>85.89</td>
      <td>562.1</td>
      <td>0.07517</td>
      <td>0.04726</td>
      <td>0.01271</td>
      <td>0.01117</td>
      <td>0.1421</td>
      <td>0.05763</td>
      <td>...</td>
      <td>14.69</td>
      <td>35.63</td>
      <td>97.11</td>
      <td>680.6</td>
      <td>0.1108</td>
      <td>0.1457</td>
      <td>0.07934</td>
      <td>0.05781</td>
      <td>0.2694</td>
      <td>0.07061</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>173</th>
      <td>11.08</td>
      <td>14.71</td>
      <td>70.21</td>
      <td>372.7</td>
      <td>0.10060</td>
      <td>0.05743</td>
      <td>0.02363</td>
      <td>0.02583</td>
      <td>0.1566</td>
      <td>0.06669</td>
      <td>...</td>
      <td>11.35</td>
      <td>16.82</td>
      <td>72.01</td>
      <td>396.5</td>
      <td>0.1216</td>
      <td>0.0824</td>
      <td>0.03938</td>
      <td>0.04306</td>
      <td>0.1902</td>
      <td>0.07313</td>
    </tr>
    <tr>
      <th>134</th>
      <td>18.45</td>
      <td>21.91</td>
      <td>120.20</td>
      <td>1075.0</td>
      <td>0.09430</td>
      <td>0.09709</td>
      <td>0.11530</td>
      <td>0.06847</td>
      <td>0.1692</td>
      <td>0.05727</td>
      <td>...</td>
      <td>22.52</td>
      <td>31.39</td>
      <td>145.60</td>
      <td>1590.0</td>
      <td>0.1465</td>
      <td>0.2275</td>
      <td>0.39650</td>
      <td>0.13790</td>
      <td>0.3109</td>
      <td>0.07610</td>
    </tr>
    <tr>
      <th>371</th>
      <td>15.19</td>
      <td>13.21</td>
      <td>97.65</td>
      <td>711.8</td>
      <td>0.07963</td>
      <td>0.06934</td>
      <td>0.03393</td>
      <td>0.02657</td>
      <td>0.1721</td>
      <td>0.05544</td>
      <td>...</td>
      <td>16.20</td>
      <td>15.73</td>
      <td>104.50</td>
      <td>819.1</td>
      <td>0.1126</td>
      <td>0.1737</td>
      <td>0.13620</td>
      <td>0.08178</td>
      <td>0.2487</td>
      <td>0.06766</td>
    </tr>
    <tr>
      <th>317</th>
      <td>18.22</td>
      <td>18.87</td>
      <td>118.70</td>
      <td>1027.0</td>
      <td>0.09746</td>
      <td>0.11170</td>
      <td>0.11300</td>
      <td>0.07950</td>
      <td>0.1807</td>
      <td>0.05664</td>
      <td>...</td>
      <td>21.84</td>
      <td>25.00</td>
      <td>140.90</td>
      <td>1485.0</td>
      <td>0.1434</td>
      <td>0.2763</td>
      <td>0.38530</td>
      <td>0.17760</td>
      <td>0.2812</td>
      <td>0.08198</td>
    </tr>
    <tr>
      <th>138</th>
      <td>14.95</td>
      <td>17.57</td>
      <td>96.85</td>
      <td>678.1</td>
      <td>0.11670</td>
      <td>0.13050</td>
      <td>0.15390</td>
      <td>0.08624</td>
      <td>0.1957</td>
      <td>0.06216</td>
      <td>...</td>
      <td>18.55</td>
      <td>21.43</td>
      <td>121.40</td>
      <td>971.4</td>
      <td>0.1411</td>
      <td>0.2164</td>
      <td>0.33550</td>
      <td>0.16670</td>
      <td>0.3414</td>
      <td>0.07147</td>
    </tr>
  </tbody>
</table>
<p>188 rows × 30 columns</p>
</div>

We have 188 observations for testing purposes. Now for the exciting part, let's train our model and see how it performs in classifying malignant breast cancer cases. 

## Decision tree classifier
First, import the *decisiontreeclassifer* algorithm. Here you can set differet stopping criteria to prevent the tree from becoming bulky. In this case, the *ccp_alpha* library is used to perform pruning. You can read more [here](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py)

```python
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(ccp_alpha=0.01)
clf = clf.fit(X_train, y_train)
```
Below is how the model is classifying each observation in the test dataframe *'0' is for malignant* and *'1' is for benign*
```python
predictions = clf.predict(X_test)
predictions
```
array([0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0,
       1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0,
       1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1,
       0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1,
       0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1,
       1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0,
       1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1,
       1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0])

The probability of each classification can be obtained with the command below.
```
clf.predict_proba(X_test)
```
## Visualizing the decision tree 
```
from sklearn import tree
from matplotlib import pyplot as plt
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf, 
                   feature_names=feature_names,  
                   class_names={0:'Malignant', 1:'Benign'},
                   filled=True,
                  fontsize=12)
plt.show()
```
![Decision tree classifier for breast cancer](file:///c%3A/Users/USER/output2.png)


## Model fit and accuracy
The accuracy of the model is 94.7% 
```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)
```
```
0.9468085106382979
```
Precision, recall and f1-value are all more than 90%
```python
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions, target_names=['malignant', 'benign']))
```
```
              precision    recall  f1-score   support

   malignant       0.92      0.95      0.93        73
      benign       0.96      0.95      0.96       115

    accuracy                           0.95       188
   macro avg       0.94      0.95      0.94       188
weighted avg       0.95      0.95      0.95       188

```
## Which attributes are important in classifying breast cancer? 

We have 30 attributes in the dataset inorder to determine which features are the most important in classifying breast cancer malignancy let's run the '*clf_feature_importances_*' function 

```python
feature_names = X.columns
feature_importance = pd.DataFrame(clf.feature_importances_, index = feature_names).sort_values(0, ascending=False)
feature_importance
feature_importance.head(10).plot(kind='bar')
```
![Importance of the attributes](file:///c%3A/Users/USER/output1.png)

# Key takeaways

The decision tree classifier has identified the best features/attributes to classify a case of breast tumor into benign and malignant. Using these features, the model the classify cases with 94.7% accuracy and 95% precision. 
# Conclusion 
Decision trees are simple and flexible algorithms that are easy to interpret. Especially, since feature importance can be identified it avoids the commonly faced **black box** problem with neural networks. However, decision trees are prone to overfitting (which can be addressed with pruning) and a small variation within data can produce a significant change. 



