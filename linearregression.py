from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder

from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import numpy as np


df=pd.read_csv(r"C:\Users\guang\Desktop\study5be\ml\personal\1\public_dataset\train_2.csv")
pred_df=pd.read_csv(r"C:\Users\guang\Desktop\study5be\ml\personal\1\public_dataset\test_sample.csv")

df.dropna()

#模型初评估
lm = ols('charges ~ age + children',data=df).fit()
print(lm.summary())

#df.info()
#pred_df.info()

#print(df.describe())
#本来打算用热编码，但是pandas的get_dummies函数实在是好用，效果一样
#le_sex=OneHotEncoder()
#le_region=OneHotEncoder()
#le_smoker=OneHotEncoder()

#df['sex']=le_sex.fit_transform(df['sex'])
#df['region']=le_region.fit_transform(df['region'])
#df['smoker']=le_smoker.fit_transform(df['smoker'])

dummies = pd.get_dummies(df,drop_first = True)
dummies.info()
#print (dummies.duplicated())
pred_dummies = pd.get_dummies(pred_df,drop_first = True)
#pred_dummies.info()
#print(pred_dummies.head())
#pred_df['sex']=le_sex.fit_transform(pred_df['sex'])
#pred_df['region']=le_region.fit_transform(pred_df['region'])
#pred_df['smoker']=le_smoker.fit_transform(pred_df['smoker'])
#模型后评估


plt.title(u'san dian tu')
plt.xlabel('smoker_yes')
plt.ylabel('charges')
plt.legend()
plt.scatter(smoker_yes_value,charges_value,s=20,c="#ff1212",marker='o')
plt.show()
plt.xlabel('age')
plt.ylabel('charges')
plt.legend()
plt.scatter(age_value,charges_value,s=20,c="#ff1212",marker='o')
plt.show()
plt.xlabel('bmi')
plt.ylabel('charges')
plt.legend()
plt.scatter(bmi_value,charges_value,s=20,c="#ff1212",marker='o')
plt.show()
plt.xlabel('children')
plt.ylabel('charges')
plt.legend()
plt.scatter(children_value,charges_value,s=20,c="#ff1212",marker='o')
plt.show()
plt.xlabel('region_southwest')
plt.ylabel('charges')
plt.legend()
plt.scatter(region_southwest_value,charges_value,s=20,c="#ff1212",marker='o')
plt.show()
plt.xlabel('region_southeast')
plt.ylabel('charges')
plt.legend()
plt.scatter(region_southeast_value,charges_value,s=20,c="#ff1212",marker='o')
plt.show()


age_value=dummies['age'].values
bmi_value=dummies['bmi'].values
children_value=dummies['children'].values
smoker_yes_value=dummies['smoker_yes'].values
region_southwest_value=dummies['region_southwest'].values
region_southeast_value=dummies['region_southeast'].values
charges_value=dummies['charges'].values


#for i in range(0,1070):
#   if smoker_yes_value[i]==1:
#       if charges_value[i] > 72000:
#           print(i)

#更新了数据，改用删除数据后的train_2数据，并且根据散点图，把bmi属性从线性模型中删除




lm=ols('charges ~ age + children + bmi + smoker_yes + sex_male + region_northwest + region_southwest + region_southeast',data=dummies).fit()
print(lm.summary())


def vif(df, col_i):
    """
    df: 整份数据
    col_i：被检测的列名
    """
    cols = list(df.columns)
    cols.remove(col_i)
    cols_noti = cols
    formula = col_i + '~' + '+'.join(cols_noti)
    r2 = ols(formula, df).fit().rsquared
    return 1. / (1. - r2)

#test_data = dummies[['age', 'children', 'bmi', 'smoker_yes', 'sex_male','region_northwest','region_southwest','region_southeast']]
#for i in test_data.columns:
#    print(i, '\t', vif(df=test_data, col_i=i))
print(lm.params)


re = open(r"C:\Users\guang\Desktop\re.txt",'w')
#print(pred_dummies.size)
#print(pred_dummies.shape)



for i in range(0,268):
    result = -13010.31 + 304.43*float(pred_dummies['age'].values[i]) + 627.32*float(pred_dummies['children'].values[i]) + 481.01*float(pred_dummies['bmi'].values[i]) + 28806.20*float(pred_dummies['smoker_yes'].values[i]) -28.14*float(pred_dummies['sex_male'].values[i]) - 300.89*float(pred_dummies['region_northwest'].values[i]) - 1119.00*float(pred_dummies['region_southwest'].values[i]) - 1476.24*float(pred_dummies['region_southeast'].values[i])
    re.write(str(result))
    re.write("\n")

re.close()
