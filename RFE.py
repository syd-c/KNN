import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFE
import time
from xgboost import XGBRegressor
from math import sqrt

start = time.time()####计时开始
########################## 读入数据并分为trainSet和testSet #################################
data = pd.read_excel(r'C:\Users\Ad\Desktop/613合并形成能.xlsx',
                     sheet_name='CE1-C-B', index_col=None, header=[0], usecols=None)
# regr_RF = RandomForestRegressor(n_estimators=9, max_depth=8,random_state=0)
regr_RF = RandomForestClassifier(n_estimators=17, max_depth=7,random_state=0)
model = regr_RF####RFE中的模型
# 删除有缺失值的行
# data.dropna(inplace=True)

# 将数据分成X和y
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]
print(X, y)
# 将数据缩放至[0, 1]间。训练过程: fit_transform()
# std = MinMaxScaler()
# X = DataFrame(std.fit_transform(X))
X_lie = X.shape[1]
print('特征集的列数=', X.shape[1])#X.shape[1]列数，X.shape[0]行数
# data.to_excel("path",sheet_name='Standardization')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# random_state:随机种子。这个东西是会根据你填的数字多少它对最终的数据结果是有影响的，如果你每次都填1，
# 其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样


################################### 选择模型 ############################################
# regr_KR = KernelRidge(alpha=.001, kernel='rbf',gamma=2**-7)
# regr_DT = tree.DecisionTreeRegressor(max_depth=5)       ###max_depth相同结果也有所区别

# regr_MLP = MLPRegressor(random_state=1, max_iter=500)
# regr_XGBR = XGBRegressor(booster='gbtree',max_depth=5,min_child_weight=1,gamma=0,reg_alpha=0.1,learning_rate=0.3,random_state=0)
####
# LR= LinearRegression()
# # 挑选出7个相关的变量
# rfe_model = RFE(model, 7)
# # 交给模型去进行拟合
# X_rfe = rfe_model.fit_transform(X,y)
# LR.fit(X_rfe,y)
# # 输出各个变量是否是相关的，并且对其进行排序
# print(rfe_model.support_)
# print(rfe_model.ranking_)


# 将13个特征变量都依次遍历一遍
feature_num_list = np.arange(1, X_lie+1)
print(feature_num_list)
# 定义一个准确率
high_score = 0
# 最优需要多少个特征变量
num_of_features = 0
score_list_train = []
MAE_list_train = []
RMSE_list_train=[]
score_list = []
MAE_list = []
RMSE_list=[]

AUC_train_list = []
AUC_list =[]
for n in range(0, len(feature_num_list)):###上面的列表是1-140，但是list[0]=1,所以从0开始倒139
    # X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    # model = RandomForestRegressor()
    rfe_model = RFE(model, n_features_to_select=feature_num_list[n], step=1)
    X_train_rfe_model = rfe_model.fit_transform(X_train, y_train)
    X_test_rfe_model = rfe_model.transform(X_test)
    model.fit(X_train_rfe_model, y_train)

    y_train_model = model.predict(X_train_rfe_model)
    y_test_model = model.predict(X_test_rfe_model)

    score_train = model.score(X_train_rfe_model, y_train)
    score = model.score(X_test_rfe_model, y_test)
    # print('X_test_rfe_model, y_test', X_test_rfe_model, y_test)
    score_list_train.append(score_train)
    score_list.append(score)

    # MAE_train=np.mean(abs(y_train_model-y_train))
    # MAE = np.mean(abs(y_test_model-y_test))
    # MAE_list_train.append(MAE_train)
    # MAE_list.append(MAE)
    # RMSE_train=sqrt(mean_squared_error(y_train, y_train_model))
    # RSME = sqrt(mean_squared_error(y_test, y_test_model))
    # RMSE_list_train.append(RMSE_train)
    # RMSE_list.append(RSME)
    AUC_train = roc_auc_score(y_train_model, y_train)
    AUC_train_list.append(AUC_train)
    AUC = roc_auc_score(y_test_model, y_test)  # AUC###RFC
    AUC_list.append(AUC)###RFC
    if score > high_score:
        high_score = score
        num_of_features = feature_num_list[n]
print('score_train_list=', score_list_train)
# print('MAE_train_list=', MAE_list_train)
# print('RMSE_train_list=', RMSE_list_train)
print('score_test_list=', score_list)
# print('MAE_test_list=', MAE_list)
# print('RMSE_test_list=', RMSE_list)
# print("最优的变量是: %d个" % num_of_features)
# print("%d个变量的准确率为: %f" % (num_of_features, high_score))
print('AUC_train_list=', AUC_train_list)###RFC
print('AUC_test_list=', AUC_list)###RFC
# num_of_features = 9 #单独筛选时需要手动输入该变量
cols = list(X.columns)
# model = RandomForestRegressor()
# 初始化RFE模型，筛选出10个变量
rfe_model_best = RFE(model, n_features_to_select=num_of_features)##原来X_rfe = rfe.fit_transform(X,y)！！因为上面改成了训练集
X_rfe = rfe_model_best.fit_transform(X_train, y_train)
# 拟合训练模型
model.fit(X_rfe, y_train)###原来model.fit(X_rfe,y)
df = pd.Series(rfe_model_best.support_, index=cols)
selected_features = df[df==True].index
print('筛选出来最优的特征是:', selected_features)#####跟特征名称差1,selected_features+1

end = time.time()
print('运行时间=', end-start)

# f = open(r'RFE.txt','w')
#
# print('score_list= ', score_list, '\n'
#         'MAE_list= ', MAE_list, '\n'
#         'RMSE_list=', RMSE_list, '\n'
#         "最优的变量是: %d个" % num_of_features,'\n'
#         "最优的变量是: %d个" % num_of_features,'\n'
#         "%d个变量的准确率为: %f" % (num_of_features, high_score), '\n',
#         '筛选出来最优的特征是:', selected_features,
#       '运行时间=', end-start, '\n',
#       file=f)
#
# f.close()
