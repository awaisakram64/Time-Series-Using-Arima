"""
Created on Sun Sep  1 20:35:41 2019

@author: awais
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import matplotlib


def train_model(train):
        
    train = train[["date", "y"]]
    train.columns = ["date", "y"]
    train.date = pd.to_datetime(train.date)
    
    # aggregate data on days
    df = train.groupby('date')['y'].sum().reset_index()
    
    df = df.set_index('date')
    
    '''
    ARIMA models are denoted with the notation ARIMA(p, d, q). 
    These three parameters account for seasonality, trend, and noise in data
    '''
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    
    
    lis0 = [] # Save aic over the iterations 
    lis1 = [] # Save pdq params for all iterations 
    lis2 = [] # Save seasonal param in all iterations
    
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(df,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                print('...')
                results = mod.fit()
                lis0.append(results.aic)
                lis1.append(param)
                lis2.append(param_seasonal)
            except:
                continue
    
    # minimum index of aic
    min_aic_index = lis0.index(min(lis0))
    order = lis1[min_aic_index]
    seasonal_order = lis2[min_aic_index]
    

    '''
    selecting the minimun aic order for forecasting
    #Fitting the ARIMA model
    '''
    mod = sm.tsa.statespace.SARIMAX(df,
                                    order=order,
                                    seasonal_order=seasonal_order,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    return results




'''
dates_to_predict = pd.date_range(start= pd.to_datetime(start_date), end= pd.to_datetime(end_date))

prediction = pd.DataFrame({'date':dates_to_predict, 'y':np.zeros(len(dates_to_predict))})

dates_to_predict.shape
prediction.dtypes

# concatinate dataframe of zero trans with transection data
merg_data = pd.concat([df.reset_index()[['date', 'y']], prediction[['date', 'y']]])
merg_data = merg_data.set_index('date')
'''

def predictions(df, model, start_date, end_date):    
    pred = model.get_prediction(start=pd.to_datetime(start_date),end=pd.to_datetime(end_date), dynamic=False)
    pred_ci = pred.conf_int()
    
    ax = df.plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    plt.legend()
    plt.show()
    return pred.predicted_mean

train = pd.read_csv('train.csv')
train = train[["Date", "y"]]    
train.columns = ["date", "y"]
train.date = pd.to_datetime(train.date)

# aggregate data on days
df = train.groupby('date')['y'].sum().reset_index()
df = df.set_index('date')

print("example date")
print("start_date = 2019-07-01   end_date    = 2019-08-31 ")

start_date = '2019-07-01'
end_date    = '2019-08-31'

model = train_model(train)

prediction_n_plots = predictions(df, model, start_date, end_date)
