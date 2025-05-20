from pandas import DataFrame
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import time
from sklearn.model_selection import cross_val_score

start = time.time()
data = pd.read_excel(r'C:/Users/Ad/Desktop/613合并形成能.xlsx',
                     sheet_name='CE3', index_col=None, header=[0], usecols=None)


X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]
print(X, y)

std = MinMaxScaler()
X = DataFrame(std.fit_transform(X))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# regr_KR = KernelRidge(alpha=.001, kernel='rbf')
# regr_DT = tree.DecisionTreeRegressor(max_depth=5)      
regr_RF = RandomForestRegressor(random_state=0)
# regr_RF = RandomForestClassifier(random_state=0)
# regr_MLP = MLPRegressor(random_state=1, max_iter=500)

# # 3.RandomForestRegressor
# regr_RF.fit(X_train, y_train)
# y_test_pred_RF = regr_RF.predict(X_test)
# y_train_pred_RF = regr_RF.predict(X_train)
# print('3.RandomForestRegressor:')
# print('-' * 30)
# print("train_RMSE: %.3f" % sqrt(mean_squared_error(y_train, y_train_pred_RF)))
# print('train_MAE: %.3f' % np.mean(abs(y_train - y_train_pred_RF)))
# print('train_R2: %.3f' % r2_score(y_train, y_train_pred_RF))
# print("test_RMSE: %.3f"%sqrt(mean_squared_error(y_test, y_test_pred_RF)))
# print('test_MAE: %.3f' % np.mean(abs(y_test-y_test_pred_RF)))
# print('test_R2: %.3f' % r2_score(y_test, y_test_pred_RF))
# print('-'*30)



# param = {"n_estimators": range(1, 201,1),'max_depth': range(1, 31)}
# # 'max_depth': range(1, 100)
# ###
# gsearch = GridSearchCV(estimator=regr_RF, param_grid=param, cv=3)#cv=
# # gsearch.fit(X_train, y_train)
# gsearch.fit(X, y)
#
# # print(gsearch.grid_scores_)
#
# print("best params")
# print(gsearch.best_params_)
# print("best scores")
# print(gsearch.best_score_)


scorel_train = []
scorel_test  = []
for n in range(0, 200, 1):
    # regr_RF = RandomForestClassifier(n_estimators=i + 1,
    #                                 n_jobs=-1,
    #                                 random_state=0)

    regr_RF = RandomForestRegressor(n_estimators=n+1,
                                 n_jobs=-1,
                                 random_state=0)
    regr_RF.fit(X_train, y_train)
    y_test_pred_RF = regr_RF.predict(X_test)
    y_train_pred_RF = regr_RF.predict(X_train)
    score_train = r2_score(y_train, y_train_pred_RF)
    score_test  = r2_score(y_test, y_test_pred_RF)
    scorel_train.append(score_train)
    scorel_test.append(score_test)
#
#
# print('scorel_train =',scorel_train)
# print(len(scorel_train))
# print('scorel_test =', scorel_test)
# print(len(scorel_test))
# # print(max(scorel_train),(scorel_train.index(max(scorel_train))*10)+1)
# print(max(scorel_train), (scorel_train.index(max(scorel_train)))+1)
# print(max(scorel_test), (scorel_test.index(max(scorel_test)))+1)


scorel_train = []
scorel_test  = []
for md in range(1, 41, 1):
    # regr_RF = RandomForestClassifier(n_estimators=86,max_depth=md,
    #                              n_jobs=-1,
    #                              random_state=0)
    regr_RF = RandomForestRegressor(n_estimators=9,max_depth=md,
                                 n_jobs=-1,
                                 random_state=0)
    regr_RF.fit(X_train, y_train)
    y_test_pred_RF = regr_RF.predict(X_test)
    y_train_pred_RF = regr_RF.predict(X_train)
    score_train = r2_score(y_train, y_train_pred_RF)
    score_test  = r2_score(y_test, y_test_pred_RF)
    scorel_train.append(score_train)
    scorel_test.append(score_test)


print('scorel_train =',scorel_train)
print(len(scorel_train))
print('scorel_test =', scorel_test)
print(len(scorel_test))
# print(max(scorel_train),(scorel_train.index(max(scorel_train))*10)+1)
print(max(scorel_train), (scorel_train.index(max(scorel_train)))+1)
print(max(scorel_test), (scorel_test.index(max(scorel_test)))+1)



# plt.figure('CE1-train+test-n_estimators', figsize=[5, 5], dpi=600)
# plt.rcParams['font.sans-serif'] = ['Times New Roman']
# # plt.plot(range(1, 201, 1), scorel_train,color=[1,.5,.055],linewidth=0.5, label='TrainSet')
# # plt.plot(range(1, 201, 1), scorel_test,color=[.12,.47,.7],linewidth=0.5, label='TestSet')
# plt.plot(range(1, 41, 1), scorel_train,color=[1,.5,.055],linewidth=0.5, label='TrainSet')
# plt.plot(range(1, 41, 1), scorel_test,color=[.12,.47,.7],linewidth=0.5, label='TestSet')
# plt.xticks(fontsize=4, weight='bold')
# plt.yticks(fontsize=4, weight='bold')
# # plt.xlabel('n_estimators', family='Times New Roman', fontsize=5, weight='bold')
# plt.xlabel('max_depth', family='Times New Roman', fontsize=5, weight='bold')
# plt.ylabel('R-square', family='Times New Roman', fontsize=5, weight='bold')
# plt.grid(True)

# plt.grid(color='black', linestyle='-', linewidth=0.5, alpha=0.1)
# ax=plt.gca()#plt.plot(x,x)
# ax.locator_params('x',nbins=20)
# ax.locator_params('y',nbins=10)
# plt.legend(prop={'family':'Times New Roman','size':4},loc="lower right")
# plt.show()
#
# #

# # scorel = []
# # for i in range(0, 200, 1):
# #     rfr = RandomForestRegressor(n_estimators=i+1,
# #                                  n_jobs=-1,
# #                                  random_state=0)
# #     score = cross_val_score(rfr, X, y, cv=5).mean()
# #     scorel.append(score)
# #
# # # print(max(scorel), (scorel.index(max(scorel))*10)+1)###*10是for i in range(0, 200, 10)
# # print(max(scorel), (scorel.index(max(scorel)))+1)
# # plt.figure(figsize=[10, 5])
# # plt.plot(range(1, 201, 1), scorel)
# # plt.show()
#
# print('n=100 R2',scorel_test[99])
end = time.time()
print('run_time=', end-start)
