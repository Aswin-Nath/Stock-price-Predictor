#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np
import yfinance as yf
import math
import matplotlib.pyplot as plt


# In[27]:


from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[28]:


file=yf.download("TCS",start="2010-01-01",end="2022-12-31")


# In[29]:


file


# In[30]:


file.shape


# In[31]:


close=pd.DataFrame(file["Close"])


# In[32]:


close.mean()


# In[33]:


close


# In[34]:


plt.plot(np.array(close.index),np.array(close["Close"]))


# In[35]:


length=int(len(close)*0.9)


# In[36]:


training_data=list(close[:length]["Close"])
testing_data=list(close[length:]["Close"])


# In[37]:


plt.figure(figsize=(15,7))
plt.grid(True)
plt.xlabel("Dates")
plt.ylabel("Closing prices")
plt.plot(close[:length]["Close"],"green",label="Train data")
plt.plot(close[length:]["Close"],"b",label="Test Data")


# In[38]:


n_test=len(testing_data)
model_predictions=[]


# In[39]:


import statsmodels.api as sm
for i in range(n_test):
  model = sm.tsa.ARIMA(training_data, order=(4,1,0))
  model_fit=model.fit()
  output=model_fit.forecast()
  yhat=output[0]
  model_predictions.append(yhat)
  actual=testing_data[i]
  training_data.append(actual)


# In[44]:


print(model_fit.summary())


# In[41]:


plt.figure(figsize=(15,10))
plt.grid(1)
date_range=np.array(close[length:].index)
plt.plot(date_range,np.array(model_predictions),"b",marker="o",linestyle="dashed",label="STOCK PREDICTION PRICE")
plt.plot(date_range,np.array(testing_data),"r",label="STOCK ACTUAL PRICE")
plt.xlabel("DATE")
plt.ylabel("PRICE")
plt.legend()
plt.show()


# In[42]:


MSE=np.mean(np.abs(np.array(model_predictions)-np.array(testing_data))/np.abs(testing_data))


# In[43]:


100-MSE


# In[ ]:




