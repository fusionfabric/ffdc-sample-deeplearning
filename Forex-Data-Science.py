
#API Handling Libraries -- requests, re, json
import requests
import re
import json 

#Data Handling Libraries -- numpy, pandas, datetime
import numpy as np
import pandas as pd
import datetime
from datetime import date
from pandas_datareader import data

#Visualization Libraries -- matplotlib, seaborn, and pygal
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
import seaborn as sns
import pygal

#Machine Learning Libraries -- sklearn and keras
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, LSTM, Dense


# Authentication Token:
class authenticate:
    '''
    config_file: A csv that stores the client id, client secret, reply url, token url and base url
    '''
    def __init__(self,config_file):
        self.df = pd.read_csv(config_file)
        self.client_id = self.df['application_id'].values[0]
        self.client_secret = self.df['access_key'].values[0]
        self.reply_url = self.df['reply_url'].values[0]
        self.token_endpoint = self.df['token_url'].values[0]
        self.base_url = self.df['base_url'].values[0]

    def get_token(self):
        headers = {
        }
        data = {
            'grant_type'     : 'client_credentials',
            'client_id'      : self.client_id,
            'client_secret'  : self.client_secret
        }
        response = requests.request('POST', self.token_endpoint, data=data, headers=headers)
        cleaned_response = response.text.replace(':null', ':"null"')
        response_dict = eval(cleaned_response)
        token = response_dict['access_token']
        return response, token

a = authenticate('config.csv')
response,token = a.get_token()

# Get Foreign Exchange Data:
class tcm_api:
    def __init__(self, config_file, token, method, endpoint, payload, additional_headers={}, params=None):
        self.df = pd.read_csv(config_file)
        self.base_url = self.df['base_url'].values[0]
        self.token = token
        self.endpoint_type = method
        self.endpoint = "/fxrate/v1/"+endpoint
        self.additional_headers = additional_headers
        self.params = params

    def connect_endpoint(self):
        self.url = self.base_url + self.endpoint
        headers = {
            'Authorization': 'Bearer ' + self.token,
            'Content-Type': 'application/json',
        }
        for key in self.additional_headers.keys():
            headers[key] = self.additional_headers[key]
        stringified_payload = json.dumps(payload)
        response = requests.request(self.endpoint_type, self.url, params=self.params, data=stringified_payload,
                                    headers=headers)
        return response

#Example GET Request for Forex Data on 2019-02-01
method = 'GET'
endpoint = '2019-02-01'
payload = ''
additional_headers = {}
params = {}
config_file = 'config.csv'
a = authenticate(config_file)
# response,token = a.get_token()
tcm = tcm_api(config_file, token, method, endpoint, payload, additional_headers, params)
response = tcm.connect_endpoint()

# Generate Dates to call the API with:
def generate_dates(numdays):
    #Get todays date and all number of numdays previous and convert them to a string in Year-Month-Day Format
    base = datetime.datetime.today()
    date_list = [(base - datetime.timedelta(days=x)).strftime("%Y-%m-%d") for x in range(0, numdays)]
    return date_list

years = 20
dates = generate_dates(numdays=365*years)
print("Dates Sample: ", dates[0:10])

# Get 20 years of EUR/USD data:
#Set API Request parameters
method = 'GET'
payload = ''
config_file = 'config.csv'
additional_headers = {}
params = {}
#Iterate over all dates to get fx rate for that day
currencies_data = []

print("Retrieving data from the API server. Please be patient, there are 20 years of data!")
for idx,date in enumerate(dates):
    endpoint = date
    #Re-Authenticate every 50 requests
    if idx % 50 == 0:
        a = authenticate(config_file)
        response,token = a.get_token()
    #Make the request
    tcm = tcm_api(config_file,token,method,endpoint,payload,additional_headers,params)
    response = tcm.connect_endpoint()
    #Extract the EUR/USD rate and add it to list as [ [rate,weekday/Weekend,date]]
    dta = eval(response.text)
    usd = dta["rates"]["USD"]
    ls = [usd]
    print(dta)
    if dta["date"] != date:
        ls.append("Weekend")
    else:
        ls.append("Weekday")
    ls.append(date)
    currencies_data.append(ls)
#Create DataFrame from list of list data
df = pd.DataFrame(currencies_data)
df = df.rename(columns={0:'EUR/USD',1:'PartOfWeek',2:'Date'})
#Save DataFrame to CSV for later use
df.to_csv('EUR_USD_20_years.csv',index=None)
print("All data retrieved and saved to disk: EUR_USD_20_years.csv")

# Load Saved Data:
df = pd.read_csv('EUR_USD_20_years.csv')

# Reorder and display Data:
df_clean = df.iloc[::-1]
df_clean.head(20)

# Generate Features:
def add_datepart(df, fldname, drop=True):
    '''
    A function to calculate some date features
    '''
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'):
        df[targ_pre+n] = getattr(fld.dt,n.lower())
    df[targ_pre+'Elapsed'] = fld.astype(np.int64) // 10**9
    if drop: df.drop(fldname, axis=1, inplace=True)

#Generate Date Features
add_datepart(df_clean, 'Date')

#Generate Technical indicators (Simple/Exponential Moving Averages)
df_clean['sma5'] = df_clean['EUR/USD'].rolling(5).mean()
df_clean['sma2'] = df_clean['EUR/USD'].rolling(2).mean()
df_clean['sma3'] = df_clean['EUR/USD'].rolling(3).mean()
df_clean['ema5'] = df_clean['EUR/USD'].ewm(span=5,min_periods=0,adjust=False,ignore_na=False).mean()
df_clean['ema2'] = df_clean['EUR/USD'].ewm(span=2,min_periods=0,adjust=False,ignore_na=False).mean()
df_clean['ema3'] = df_clean['EUR/USD'].ewm(span=3,min_periods=0,adjust=False,ignore_na=False).mean()

#Remove first 5 data points since will contain NaN from SMA5/EMA5
df_clean = df_clean.iloc[5:]
keep_cols = list(df_clean.columns.values)
keep_cols.remove('PartOfWeek')
df_feat = df_clean[keep_cols]
df_clean.head()

# Feature Analysis:
colormap = plt.cm.RdPu
plt.figure(figsize=(15,15))
plt.title('Pearson correlation of features', y=1.05, size=15)
sns.heatmap(df_clean.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()

# Create Input and Target Data:
#Set number of days to predict ahead and an option to scale our data
future_prediction_days = 1
scale_data = True

#Create Forward column that has the shifted price
df_feat['Forward'] = df_feat['EUR/USD'].shift(-1 * future_prediction_days)
df_feat = df_feat[:-1 * future_prediction_days]

#Create Direction column to get binary shifted price direction
def get_direction(row1, row2):
    if row2 >= row1:
        return 1
    else:
        return 0

df_feat['Direction'] = df_feat[['EUR/USD', 'Forward']].apply(lambda i: get_direction(i[0], i[1]), axis=1)
target_index = df_feat.columns.tolist().index('Forward')
dataset = df_feat.values.astype('float32')

#Option to scale data
if scale_data:
    # Scale the data between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    y_scaler = MinMaxScaler(feature_range=(0, 1))
    t_y = df_feat['EUR/USD'].values.astype('float32')
    t_y = np.reshape(t_y, (-1, 1))
    y_scaler = y_scaler.fit(t_y)

#Separate the Input from the Target
X = dataset[:, :-2]
y = dataset[:, target_index]

# Feature Analysis against Target:
plt.figure(figsize=(15,5))
corr = df_feat.iloc[:,:-1].corr()
sns.heatmap(corr[corr.index == 'Forward'], linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True);

# Feature Importance:
#Build and Train a Random Forest with 100 estimators
forest = RandomForestRegressor(n_estimators = 100)
forest = forest.fit(X, y)

#Extract the feature column names and feature importances for the columns
column_list = df_feat.columns.values.tolist()
importances = forest.feature_importances_

#Zip the column names and importances togther and then sort them
col_imp = list(zip(column_list,importances))
sorted_col_imp = sorted(col_imp,key=lambda i: i[1],reverse=True)

#Create a pygal bar chart to plot the feature importances
bc = pygal.Bar(width=500, height=300, explicit_size=True)
bc.title = 'Forex Feature Importances'
for i in range(len(importances)):
    bc.add(sorted_col_imp[i][0], sorted_col_imp[i][1])
bc.render_in_browser()

# Train/Test Split:
#Set the percentage for training data
pct_train = .8

#Get the X,Y train/test sets
X_train = X[0:int(len(X)*pct_train)]
y_train = y[0:int(len(y)*pct_train)]
X_test = X[int(len(X)*pct_train):]
y_test = y[int(len(y)*pct_train):]

#Reshape our data into time series for an LSTM model
X_train = X_train.reshape((X_train.shape[0],1,20))
X_test = X_test.reshape((X_test.shape[0],1,20))

# Create Model:
model = Sequential()
model.add(LSTM(20, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(20, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(4, return_sequences=False))
model.add(Dense(4, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])

print("Model Summary: ")
print(model.summary())
print("+++++++++++++++++++++++++++++++++++++++")

# Train Neural Network:
model.fit(X_train, y_train, epochs=100, batch_size=256)

# Train Set Performance:
#Get Training MSE
mse = model.history.history['mse']

#Build Line chart to show performance
line_chart = pygal.StackedLine(fill=True,width=500, height=300, explicit_size=True,
                              x_labels_major_every=10, show_minor_x_labels=False, x_label_rotation=45)
line_chart.title = 'Training Error'
line_chart.x_labels = [str(i) for i in range(100)]
line_chart.xlabel = 'Epoch'
line_chart.add('MSE', mse)
line_chart.render_in_browser()

# Test Set Performance:
#Obtain Predictions on our Test Set
preds = model.predict(X_test)

#Get Evaluation Metrics -- MSE, MAE, R2 Score, and Explained Variance Score
print("Model evaluation metrics: ")
print("Test MSE:",mean_squared_error(y_true=y_test,y_pred=preds))
print("Test MAE:  ",mean_absolute_error(y_true=y_test,y_pred=preds))
print("Test R2 Score: ",r2_score(y_pred=preds,y_true=y_test))
print("Explained Variance Score: ",explained_variance_score(y_pred=preds,y_true=y_test))
print("+++++++++++++++++++++++++++++++++++++++")

#Convert Preds and Actual Values on Test Set to Flat List for Plotting
preds_flat = preds.flatten().tolist()
y_test_flat = y_test.flatten().tolist()

#Plot Predictions vs Actual
line_chart = pygal.Line(width=500, height=300, explicit_size=True,x_labels_major_every=150, show_minor_x_labels=False, x_label_rotation=45)
line_chart.title = 'Test Predictions vs Actual'
line_chart.x_labels = df['Date'][::-1].iloc[int(0.8*len(df)):].tolist()
line_chart.add('Preds', preds_flat)
line_chart.add('Actual',  y_test_flat)
line_chart.render_in_browser()

# Save Model:
model.save('EUR_USD_1_Day_Forecast_2.h5')
print("Model saved to disk: EUR_USD_1_Day_Forecast_2.h5")

