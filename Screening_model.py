from pandas import DataFrame
from math import sqrt
import numpy as np
import pandas as pd
from sklearn import linear_model,tree
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor,RadiusNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import RFE
import warnings
warnings.filterwarnings("ignore")####屏蔽python运行时产生的FutureWarning信息

########################## 读入数据并分为trainSet和testSet #################################
data = pd.read_excel(r'C:/Users/Ad/Desktop/613合并形成能.xlsx',
                     sheet_name='DS3', index_col=None, header=[0], usecols=None)
regr_RF =RandomForestRegressor(criterion='mse',n_estimators=27, max_depth=11,random_state=0)
#CC3TOT=27/11,CE3TOT=9/8,CE1-BAND-22/15,CE1-CHANGE-17/7n_estimators=27, max_depth=11,
#volumn--n_estimators=21, max_depth=11,
# 删除有缺失值的行
# data.dropna(inplace=True)
# 将数据分成X和y
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]
# print(X, y)
# 将数据缩放至[0, 1]间。训练过程: fit_transform()
# std = MinMaxScaler()
# X = DataFrame(std.fit_transform(X))
# X.to_excel(r'特征归一化.xlsx')
# data.to_excel("path",sheet_name='Standardization')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# X_train = DataFrame(std.fit_transform(X_train))
# X_test = DataFrame(std.fit_transform(X_test))
# print(X_train)
# print(X_test)
# random_state:随机种子。这个东西是会根据你填的数字多少它对最终的数据结果是有影响的，如果你每次都填1，
# 其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样



                                       ### 选择模型 ###考虑加入网格搜索优化超参数；是否考虑交叉验证
                                       # RF没必要进行CV
regr_LR = linear_model.LinearRegression()                                                    #1
regr_RCV = linear_model.RidgeCV(alphas=[0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 1.0, 5.0, 10.0])  #2  # RidgeCV 通过内置的 Alpha 参数的交叉验证cv=来实现岭回归
regr_Lasso = linear_model.Lasso(alpha=0.1)                                                   #3
regr_LLasso = linear_model.LassoLars(alpha=.1)                                               #4
regr_BR = linear_model.BayesianRidge()                                                       #5
regr_KR = KernelRidge(alpha=.001, kernel='rbf', gamma=2**-7)                                 #6
regr_SVR = SVR(kernel='rbf', C=100, gamma=.2)                                                #7
regr_SGDR = SGDRegressor()                                                                   #8
regr_KNR = KNeighborsRegressor(n_neighbors=1,weights='distance',p=2)                         #9
regr_RNR = RadiusNeighborsRegressor(radius=3)      ####radius对性能影响较大 CC2.4 CE2.7       #10
regr_GPR = GaussianProcessRegressor()                    #kernel=rbf,random_state=0          #11
regr_DTR = tree.DecisionTreeRegressor(max_depth=5)       ###max_depth相同结果也有所区别          #12
regr_BSVR = BaggingRegressor(base_estimator=SVR(),n_estimators=10, random_state=0)           #13
# RF = RandomForestRegressor()#n_estimators=100, max_depth=10,random_state=0                 #14
# regr_RF = RFE(estimator=RF, n_features_to_select=15, step=1)

regr_ABR = AdaBoostRegressor(random_state=0, n_estimators=100)                               #15
regr_GBR = GradientBoostingRegressor(n_estimators=100, learning_rate=0.5,
    max_depth=4, max_features='auto',loss='squared_error',subsample=1, random_state=0)#      #16
#预测总能量/atom时，CE3时GBR性能最佳，但是GBR过拟合，而且RF的性能与GBR很接近，此处仍然使用RF
# regr_GBR = RFE(estimator=GBR, n_features_to_select=15, step=1)
regr_MLPR = MLPRegressor(random_state=1, max_iter=500)                                       #17
regr_XGBR = XGBRegressor(booster='gbtree',max_depth=5,min_child_weight=1,
                         gamma=0,reg_alpha=0.1,learning_rate=0.3,random_state=0)#booster='gbtree',max_depth=5,min_child_weight=1,gamma=0,reg_alpha=0.1,learning_rate=0.3,random_state=0#
# regr_XGBR = RFE(estimator=XGBR, n_features_to_select=15, step=1)
print('LR',regr_LR.get_params())
print('RCCV',regr_RCV.get_params())
print('Lasso',regr_Lasso.get_params())
print('BR',regr_BR.get_params())
print('KR',regr_KR.get_params())
print('SVR',regr_SVR.get_params())
print('SGDR',regr_SGDR.get_params())
print('KNR',regr_KNR.get_params())
print('RNR',regr_RNR.get_params())
print('GPR',regr_GPR.get_params())
print('DT',regr_DTR.get_params())
print('BSVR',regr_BSVR.get_params())
print('RF',regr_RF.get_params())
print('ABR',regr_ABR.get_params())
print('GBR',regr_GBR.get_params())
print('XGBR',regr_XGBR.get_params())
print('MLP',regr_MLPR.get_params())

                                         ### 训练模型 ###
# 1.普通最小二乘法LinearRegression
regr_LR.fit(X_train, y_train)
y_test_pred_LR = regr_LR.predict(X_test)
y_train_pred_LR = regr_LR.predict(X_train)
print('-'*30)
print('1.LinearRegression:')
print('train_RMSE: %.6f' % sqrt(mean_squared_error(y_train, y_train_pred_LR)))
print('train_MAE: %.6f' % np.mean(abs(y_train - y_train_pred_LR)))
print('train_R2: %.6f' % r2_score(y_train, y_train_pred_LR))
print('test_RMSE: %.6f' % sqrt(mean_squared_error(y_test, y_test_pred_LR)))
print('test_MAE: %.6f' % np.mean(abs(y_test-y_test_pred_LR)))
print('test_R2: %.6f' % r2_score(y_test, y_test_pred_LR))
print('-'*30)

# 2.岭回归交叉验证RidgeCV
regr_RCV.fit(X_train, y_train)
y_test_pred_RCV = regr_RCV.predict(X_test)
y_train_pred_RCV = regr_RCV.predict(X_train)
print('-'*30)
print('2.RidgeCV:')
print('train_RMSE: %.6f' % sqrt(mean_squared_error(y_train, y_train_pred_RCV)))
print('train_MAE: %.6f' % np.mean(abs(y_train - y_train_pred_RCV)))
print('train_R2: %.6f' % r2_score(y_train, y_train_pred_RCV))
print('test_RMSE: %.6f'%sqrt(mean_squared_error(y_test, y_test_pred_RCV)))
print('test_MAE: %.6f' % np.mean(abs(y_test-y_test_pred_RCV)))
print('test_R2: %.6f' % r2_score(y_test, y_test_pred_RCV))
print('-'*30)

# 3.Lasso
regr_Lasso.fit(X_train, y_train)
y_test_pred_Lasso = regr_Lasso.predict(X_test)
y_train_pred_Lasso = regr_Lasso.predict(X_train)
print('-'*30)
print('3.Lasso:')
print('train_RMSE: %.6f' % sqrt(mean_squared_error(y_train, y_train_pred_Lasso)))
print('train_MAE: %.6f' % np.mean(abs(y_train - y_train_pred_Lasso)))
print('train_R2: %.6f' % r2_score(y_train, y_train_pred_Lasso))
print('test_RMSE: %.6f' % sqrt(mean_squared_error(y_test, y_test_pred_Lasso)))
print('test_MAE: %.6f' % np.mean(abs(y_test-y_test_pred_Lasso)))
print('test_R2: %.6f' % r2_score(y_test, y_test_pred_Lasso))
print('-'*30)

# # 4.LARS Lasso
# regr_LLasso.fit(X_train, y_train)
# y_test_pred_LLasso = regr_LLasso.predict(X_test)
# y_train_pred_LLasso = regr_LLasso.predict(X_train)
# print('-' * 30)
# print('4.LARS Lasso:')
# print('train_RMSE: %.6f' % sqrt(mean_squared_error(y_train, y_train_pred_LLasso)))
# print('train_MAE: %.6f' % np.mean(abs(y_train - y_train_pred_LLasso)))
# print('train_R2: %.6f' % r2_score(y_train, y_train_pred_LLasso))
# print('test_RMSE: %.6f' %sqrt(mean_squared_error(y_test, y_test_pred_LLasso)))
# print('test_MAE: %.6f' % np.mean(abs(y_test-y_test_pred_LLasso)))
# print('test_R2: %.6f' % r2_score(y_test, y_test_pred_LLasso))
# print('-'*30)
# # >>> reg.coef_


# 5.贝叶斯岭回归BayesianRidge
regr_BR.fit(X_train, y_train)
y_test_pred_BR = regr_BR.predict(X_test)
y_train_pred_BR = regr_BR.predict(X_train)
print('-' * 30)
print('5.BayesianRidge:')
print('train_RMSE: %.6f' % sqrt(mean_squared_error(y_train, y_train_pred_BR)))
print('train_MAE: %.6f' % np.mean(abs(y_train - y_train_pred_BR)))
print('train_R2: %.6f' % r2_score(y_train, y_train_pred_BR))
print('test_RMSE: %.6f' %sqrt(mean_squared_error(y_test, y_test_pred_BR)))
print('test_MAE: %.6f' % np.mean(abs(y_test-y_test_pred_BR)))
print('test_R2: %.6f' % r2_score(y_test, y_test_pred_BR))
print('-'*30)

# 6.KernelRidge内核岭回归
regr_KR.fit(X_train, y_train)
y_test_pred_KR = regr_KR.predict(X_test)
y_train_pred_KR = regr_KR.predict(X_train)
print('-' * 30)
print('6.KernelRidge:')
print('train_RMSE: %.6f' % sqrt(mean_squared_error(y_train, y_train_pred_KR)))
print('train_MAE: %.6f' % np.mean(abs(y_train - y_train_pred_KR)))
print('train_R2: %.6f' % r2_score(y_train, y_train_pred_KR))
print('test_RMSE: %.6f' %sqrt(mean_squared_error(y_test, y_test_pred_KR)))
print('test_MAE: %.6f' % np.mean(abs(y_test-y_test_pred_KR)))
print('test_R2: %.6f' % r2_score(y_test, y_test_pred_KR))
print('-'*30)

# 7.SVR支持向量机
regr_SVR.fit(X_train, y_train)
y_test_pred_SVR = regr_SVR.predict(X_test)
y_train_pred_SVR = regr_SVR.predict(X_train)
print('-' * 30)
print('7.SVR:')
print('train_RMSE: %.6f' % sqrt(mean_squared_error(y_train, y_train_pred_SVR)))
print('train_MAE: %.6f' % np.mean(abs(y_train - y_train_pred_SVR)))
print('train_R2: %.6f' % r2_score(y_train, y_train_pred_SVR))
print('test_RMSE: %.6f' %sqrt(mean_squared_error(y_test, y_test_pred_SVR)))
print('test_MAE: %.6f' % np.mean(abs(y_test-y_test_pred_SVR)))
print('test_R2: %.6f' % r2_score(y_test, y_test_pred_SVR))
print('-'*30)

# 8.随机梯度下降(SGD)
regr_SGDR.fit(X_train, y_train)
y_test_pred_SGDR = regr_SGDR.predict(X_test)
y_train_pred_SGDR = regr_SGDR.predict(X_train)
print('-' * 30)
print('8.SGDRegressor:')
print('train_RMSE: %.6f' % sqrt(mean_squared_error(y_train, y_train_pred_SGDR)))
print('train_MAE: %.6f' % np.mean(abs(y_train - y_train_pred_SGDR)))
print('train_R2: %.6f' % r2_score(y_train, y_train_pred_SGDR))
print('test_RMSE: %.6f' %sqrt(mean_squared_error(y_test, y_test_pred_SGDR)))
print('test_MAE: %.6f' % np.mean(abs(y_test-y_test_pred_SGDR)))
print('test_R2: %.6f' % r2_score(y_test, y_test_pred_SGDR))
print('-'*30)

# 9.最近邻回归KNeighborsRegressor/
regr_KNR.fit(X_train, y_train)
y_test_pred_KNR = regr_KNR.predict(X_test)
y_train_pred_KNR = regr_KNR.predict(X_train)
print('-' * 30)
print('9.KNeighborsRegressor:')
print('train_RMSE: %.10f' % sqrt(mean_squared_error(y_train, y_train_pred_KNR)))
print('train_MAE: %.10f' % np.mean(abs(y_train - y_train_pred_KNR)))
print('train_R2: %.10f' % r2_score(y_train, y_train_pred_KNR))
print('test_RMSE: %.10f' %sqrt(mean_squared_error(y_test, y_test_pred_KNR)))
print('test_MAE: %.10f' % np.mean(abs(y_test-y_test_pred_KNR)))
print('test_R2: %.10f' % r2_score(y_test, y_test_pred_KNR))
print('-'*30)

# # # 10.RNR
# regr_RNR.fit(X_train, y_train)
# y_test_pred_RNR = regr_RNR.predict(X_test)
# y_train_pred_RNR = regr_RNR.predict(X_train)
# print('-' * 30)
# print('10.RadiusNeighborsRegressor:')
# print('train_RMSE: %.6f' % sqrt(mean_squared_error(y_train, y_train_pred_RNR)))
# print('train_MAE: %.6f' % np.mean(abs(y_train - y_train_pred_RNR)))
# print('train_R2: %.6f' % r2_score(y_train, y_train_pred_RNR))
# print('test_RMSE: %.6f' %sqrt(mean_squared_error(y_test, y_test_pred_RNR)))
# print('test_MAE: %.6f' % np.mean(abs(y_test-y_test_pred_RNR)))
# print('test_R2: %.6f' % r2_score(y_test, y_test_pred_RNR))
# print('-'*30)

# 11.高斯过程回归GaussianProcessRegressor
regr_GPR.fit(X_train, y_train)
y_test_pred_GPR = regr_GPR.predict(X_test)
y_train_pred_GPR = regr_GPR.predict(X_train)
print('-' * 30)
print('11.GaussianProcessRegressor:')
print('train_RMSE: %.6f' % sqrt(mean_squared_error(y_train, y_train_pred_GPR)))
print('train_MAE: %.6f' % np.mean(abs(y_train - y_train_pred_GPR)))
print('train_R2: %.6f' % r2_score(y_train, y_train_pred_GPR))
print('test_RMSE: %.6f' %sqrt(mean_squared_error(y_test, y_test_pred_GPR)))
print('test_MAE: %.6f' % np.mean(abs(y_test-y_test_pred_GPR)))
print('test_R2: %.6f' % r2_score(y_test, y_test_pred_GPR))
print('-'*30)

# 12.决策树DTR
regr_DTR.fit(X_train, y_train)
y_test_pred_DTR = regr_DTR.predict(X_test)
y_train_pred_DTR = regr_DTR.predict(X_train)
print('-' * 30)
print('12.DTR:')
print('train_RMSE: %.6f' % sqrt(mean_squared_error(y_train, y_train_pred_DTR)))
print('train_MAE: %.6f' % np.mean(abs(y_train - y_train_pred_DTR)))
print('train_R2: %.6f' % r2_score(y_train, y_train_pred_DTR))
print('test_RMSE: %.6f' %sqrt(mean_squared_error(y_test, y_test_pred_DTR)))
print('test_MAE: %.6f' % np.mean(abs(y_test-y_test_pred_DTR)))
print('test_R2: %.6f' % r2_score(y_test, y_test_pred_DTR))
print('-'*30)

# 13.Bagging meta-estimator(Bagging 元-估计器)
regr_BSVR.fit(X_train, y_train)
y_test_pred_BSVR = regr_BSVR.predict(X_test)
y_train_pred_BSVR = regr_BSVR.predict(X_train)
print('-' * 30)
print('13.BaggingRegressor SVR:')
print('train_RMSE: %.6f' % sqrt(mean_squared_error(y_train, y_train_pred_BSVR)))
print('train_MAE: %.6f' % np.mean(abs(y_train - y_train_pred_BSVR)))
print('train_R2: %.6f' % r2_score(y_train, y_train_pred_BSVR))
print('test_RMSE: %.6f' %sqrt(mean_squared_error(y_test, y_test_pred_BSVR)))
print('test_MAE: %.6f' % np.mean(abs(y_test-y_test_pred_BSVR)))
print('test_R2: %.6f' % r2_score(y_test, y_test_pred_BSVR))
print('-'*30)

# 14.RandomForestRegressor
regr_RF.fit(X_train, y_train)
y_test_pred_RF = regr_RF.predict(X_test)
y_train_pred_RF = regr_RF.predict(X_train)
print('-' * 30)
print('14.RandomForestRegressor:')
print('train_RMSE: %.6f' % sqrt(mean_squared_error(y_train, y_train_pred_RF)))
print('train_MAE: %.6f' % np.mean(abs(y_train - y_train_pred_RF)))
print('train_R2: %.6f' % r2_score(y_train, y_train_pred_RF))
print('test_RMSE: %.6f' %sqrt(mean_squared_error(y_test, y_test_pred_RF)))
print('test_MAE: %.6f' % np.mean(abs(y_test-y_test_pred_RF)))
print('test_R2: %.6f' % r2_score(y_test, y_test_pred_RF))
print('-'*30)

# 15.正向激励AdaBoostRegressor
regr_ABR.fit(X_train, y_train)
y_test_pred_ABR = regr_ABR.predict(X_test)
y_train_pred_ABR = regr_ABR.predict(X_train)
print('-' * 30)
print('15.AdaBoostRegressor')
print('train_RMSE: %.6f' % sqrt(mean_squared_error(y_train, y_train_pred_ABR)))
print('train_MAE: %.6f' % np.mean(abs(y_train - y_train_pred_ABR)))
print('train_R2: %.6f' % r2_score(y_train, y_train_pred_ABR))
print('test_RMSE: %.6f' %sqrt(mean_squared_error(y_test, y_test_pred_ABR)))
print('test_MAE: %.6f' % np.mean(abs(y_test-y_test_pred_ABR)))
print('test_R2: %.6f' % r2_score(y_test, y_test_pred_ABR))
# 对于分类（CLASSIFIER）模型，SCORE函数计算的是精确度。底层是ACCURACY_SCORE。
# 对于回归（REGRESSOR）问题，SCORE函数计算的是R^2分数。底层是R2_SCORE
# print('score: %.6f' %regr_ABC.score(X, y))
print('-'*30)

# 16.梯度提升回归GradientBoostingRegressor
regr_GBR.fit(X_train, y_train)
y_test_pred_GBR = regr_GBR.predict(X_test)
y_train_pred_GBR = regr_GBR.predict(X_train)
print('-' * 30)
print('16.GradientBoostingRegressor')
print('train_RMSE: %.6f' % sqrt(mean_squared_error(y_train, y_train_pred_GBR)))
print('train_MAE: %.6f' % np.mean(abs(y_train - y_train_pred_GBR)))
print('train_R2: %.6f' % r2_score(y_train, y_train_pred_GBR))
print('test_RMSE: %.6f' %sqrt(mean_squared_error(y_test, y_test_pred_GBR)))
print('test_MAE: %.6f' % np.mean(abs(y_test-y_test_pred_GBR)))
print('test_R2: %.6f' % r2_score(y_test, y_test_pred_GBR))
# print('score: %.6f' %regr_GBR.score(X_test,y_test))
print('-'*30)

# 17.regr_XGBR
regr_XGBR.fit(X_train, y_train)
y_test_pred_XGBR = regr_XGBR.predict(X_test)
y_train_pred_XGBR = regr_XGBR.predict(X_train)
print('-' * 30)
print('17.XGBRegressor')
print('train_RMSE: %.6f' % sqrt(mean_squared_error(y_train, y_train_pred_XGBR)))
print('train_MAE: %.6f' % np.mean(abs(y_train - y_train_pred_XGBR)))
print('train_R2: %.6f' % r2_score(y_train, y_train_pred_XGBR))
print('test_RMSE: %.5f' %sqrt(mean_squared_error(y_test, y_test_pred_XGBR)))
print('test_MAE: %.5f' % np.mean(abs(y_test-y_test_pred_XGBR)))
print('test_R2: %.5f' % r2_score(y_test, y_test_pred_XGBR))
# print('score: %.6f' %regrXGBRR.score(X_test,y_test))
print('-'*30)

# 18.多层感知机(MLP)
regr_MLPR.fit(X_train, y_train)
y_test_pred_MLPR = regr_MLPR.predict(X_test)
y_train_pred_MLPR = regr_MLPR.predict(X_train)
print('-' * 30)
print('18.MLPRegressor')
print('train_RMSE: %.6f' % sqrt(mean_squared_error(y_train, y_train_pred_MLPR)))
print('train_MAE: %.6f' % np.mean(abs(y_train - y_train_pred_MLPR)))
print('train_R2: %.6f' % r2_score(y_train, y_train_pred_MLPR))
print('test_RMSE: %.6f' %sqrt(mean_squared_error(y_test, y_test_pred_MLPR)))
print('test_MAE: %.6f' % np.mean(abs(y_test-y_test_pred_MLPR)))
print('test_R2: %.6f' % r2_score(y_test, y_test_pred_MLPR))
# print('score: %.6f' %regr_MLPR.score(X_test,y_test))
print('-'*30)



####对模型准确度进行排序###
R2_dict={'LR':r2_score(y_test,y_test_pred_LR),'RCV':r2_score(y_test,y_test_pred_RCV),'Lasso':r2_score(y_test,y_test_pred_Lasso)
        ,'BR':r2_score(y_test, y_test_pred_BR),'KR':r2_score(y_test, y_test_pred_KR)
         ,'SVR':r2_score(y_test, y_test_pred_SVR),'SGD':r2_score(y_test, y_test_pred_SGDR),'KNR':r2_score(y_test, y_test_pred_KNR)
         ,'RNR':0,'GPR':r2_score(y_test, y_test_pred_GPR),'DT':r2_score(y_test, y_test_pred_DTR)
         ,'BSVR':r2_score(y_test, y_test_pred_BSVR),'RF':r2_score(y_test, y_test_pred_RF),'ABR':r2_score(y_test, y_test_pred_ABR)
         ,'GBR':r2_score(y_test, y_test_pred_GBR),'MLP':r2_score(y_test, y_test_pred_MLPR),'XGBR':r2_score(y_test, y_test_pred_XGBR)}
train_RMSE_dict={'LR':sqrt(mean_squared_error(y_train, y_train_pred_LR)),'RCV':sqrt(mean_squared_error(y_train, y_train_pred_RCV)),
           'Lasso':sqrt(mean_squared_error(y_train, y_train_pred_Lasso)),'BR':sqrt(mean_squared_error(y_train, y_train_pred_BR)),
           'KR':sqrt(mean_squared_error(y_train, y_train_pred_KR)),'SVR':sqrt(mean_squared_error(y_train, y_train_pred_SVR)),
           'SGD':sqrt(mean_squared_error(y_train, y_train_pred_SGDR)),'KNR':sqrt(mean_squared_error(y_train, y_train_pred_KNR)),
           'RNR':100,'GPR':sqrt(mean_squared_error(y_train, y_train_pred_GPR)),'DT':sqrt(mean_squared_error(y_train, y_train_pred_DTR)),
           'BSVR':sqrt(mean_squared_error(y_train, y_train_pred_BSVR)),'RF':sqrt(mean_squared_error(y_train, y_train_pred_RF)),
           'ABR':sqrt(mean_squared_error(y_train, y_train_pred_ABR)),'GBR':sqrt(mean_squared_error(y_train, y_train_pred_GBR)),
           'MLP':sqrt(mean_squared_error(y_train, y_train_pred_MLPR)),'XGBR':sqrt(mean_squared_error(y_train, y_train_pred_XGBR))}

test_RMSE_dict={'LR':sqrt(mean_squared_error(y_test, y_test_pred_LR)),'RCV':sqrt(mean_squared_error(y_test, y_test_pred_RCV)),
           'Lasso':sqrt(mean_squared_error(y_test, y_test_pred_Lasso)),'BR':sqrt(mean_squared_error(y_test, y_test_pred_BR)),
           'KR':sqrt(mean_squared_error(y_test, y_test_pred_KR)),'SVR':sqrt(mean_squared_error(y_test, y_test_pred_SVR)),
           'SGD':sqrt(mean_squared_error(y_test, y_test_pred_SGDR)),'KNR':sqrt(mean_squared_error(y_test, y_test_pred_KNR)),
           'RNR':100,'GPR':sqrt(mean_squared_error(y_test, y_test_pred_GPR)),'DT':sqrt(mean_squared_error(y_test, y_test_pred_DTR)),
           'BSVR':sqrt(mean_squared_error(y_test, y_test_pred_BSVR)),'RF':sqrt(mean_squared_error(y_test, y_test_pred_RF)),
           'ABR':sqrt(mean_squared_error(y_test, y_test_pred_ABR)),'GBR':sqrt(mean_squared_error(y_test, y_test_pred_GBR)),
           'MLP':sqrt(mean_squared_error(y_test, y_test_pred_MLPR)),'XGBR':sqrt(mean_squared_error(y_test, y_test_pred_XGBR))}

import operator
sorted_dict_r2 = sorted(R2_dict.items(), key=operator.itemgetter(1), reverse=True)###reverse=True逆序
sorted_train_dict_rmse = sorted(train_RMSE_dict.items(), key=operator.itemgetter(1), reverse=False)###reverse=True逆序
sorted_test_dict_rmse = sorted(test_RMSE_dict.items(), key=operator.itemgetter(1), reverse=False)###reverse=True逆序
print('train_RMSE',sorted_train_dict_rmse)
print('test_RMSE',sorted_test_dict_rmse)
print('R2',sorted_dict_r2)
