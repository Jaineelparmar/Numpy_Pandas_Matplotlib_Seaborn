#CREATING DATAFRAME

import pandas as pd 
weather_stuff = {
    'day':['1/1/2017', '1/2/2017', '1/3/2017', '1/4/2017', '1/5/2017', '1/6/2017'],
    'temp':[23, 34, 25, 34, 33, 12],
    'windspeed':[0, 7, 24, 3, 6, 9],
    'event':['rainy', 'sunny', 'snow', 'snow', 'rainy', 'sunny']
}
df = pd.DataFrame(weather_stuff)
print(df)


#CREATING DATAFRAME

import pandas as pd 
df = pd.read_csv('weather_stuff.csv')
print(df)
rows, columns = df.shape      #GIVES THE SIZE 
print(rows)
print(columns)
print(df.head(2))             #PRINTS TOP 2 ROWS
print(df.tail())              #PRINTS ALL ROWS EXCEPT 1ST ROW
print(df[2:5])                #PRINTS THE ROWS BETWEEN 2 AND 5
print(df.columns)             #PRINTS NAME OF COLUMNS
print(df.day) #series         #ONLY PRINTS DAY COLUMN
print(type(df.day))           #PRINTS #Series
print(type(df))               #PRINTS #Dataframe
print(df[['event', 'day']])   #two columns at a time
print(df['temperature'].max())       #finds max of temperature from all
print(df['temperature'].min())       #finds min of temperature from all
print(df['temperature'].mean())      #finds mean of temperature
print(df['temperature'].std())       #finds standard deviation of temperature
print(df.describe())                  #gives the statistics of the integer
print(df[df.temperature >= 32])         #temperature above or 32
print(df['day'][df['temperature'] == df['temperature'].max()])      #print day for max temperature
print(df[['day', 'temperature']][df['temperature'] == df['temperature'].max()])     #prints day and temperature for max temperature
print(df.index)
print(df.set_index('day', inplace = True))
In-place operation is an operation that changes directly the content of a given linear algebra, vector, matrices(Tensor) without making a copy. 
The operators which helps to do the operation is called in-place operator
df.set_index('day', inplace = True)
print(df.loc['1/2/2017'])
df.reset_index(inplace = True)
print(df)
df.set_index('event', inplace = True)
print(df.loc['Snow'])
df.reset_index(inplace = True)
print(df)



## Diff ways of Creating Dataframe

import pandas as pd 

#Csv
df = pd.read_csv('__.csv')

#Excel
df1 = pd.read_excel('___.xlsx', 'Sheet1')

#Creating dataframe from a dictionary
n = {
    'ds':[3, 4, 5],
    'sef':['ffs', 'wfw', 'efe']
}
df2 = pd.DataFrame(n)
print(df2)

#Creating dataframe from a tuples list - list contains tuples
x = [
    ('1/1/2017', 32, 6, 'Rain'),
    ('1/2/2017', 22, 16, 'sunny'),
    ('1/3/2017', 12, 26, 'Rain')
]
df3 = pd.DataFrame(x, columns = ['day', 'temp', 'windspeed', 'event'])
print(df3)

#List of dictionaries
a = [
    {'day':'1/1/2017', 'temp':23, 'windspeed':7, 'event':'Rainy'},
    {'day':'1/2/2017', 'temp':43, 'windspeed':17, 'event':'Sunny'},
    {'day':'1/3/2017', 'temp':33, 'windspeed':3, 'event':'Rainy'}
]
df4 = pd.DataFrame(a)
print(df4)


#Other Types - Google for other Types.

READ CSV AND WRITE CSV
import pandas as pd 
df = pd.read_csv('work_doc.csv')
df1 = pd.read_csv('work_doc.csv', skiprows = 1)
df2 = pd.read_csv('work_doc.csv', header = 1)
df3 = pd.read_csv('work_doc.csv', header=None)
df4 = pd.read_csv('work_doc.csv', header=None, names=['ticker','eps','revenue','price','people'])
df5 = pd.read_csv('work_doc.csv', nrows = 3)
df6 = pd.read_csv('work_doc.csv', na_values = {
            'Eps':['NOT AVAILABLE', 'n.a.'],
            'Revenue':['NOT AVAILABLE', 'n.a.', -1],
            'Price':['NOT AVAILABLE', 'n.a.'],
            'People':['NOT AVAILABLE', 'n.a.']
        })

print(df)
print(df1)
print(df2)
print(df3)
print(df4)
print(df5)
print(df6)
print(df6.to_csv('new.csv', index=False, header = False, columns=['Tickers', 'Eps']))



#READ EXCEL AND WRITE EXCEL - xlrd, openpyxl
import pandas as pd 

def convert_people_cell(cell):
    if cell == 'n.a.':
        return 'sam walton'
    return cell

def convert_eps_cell(cell):
    if cell == 'NOT AVAILABLE':
        return None
    return cell

df9  = pd.read_excel('stock_data.xlsx', 'Sheet1', converters={
        'People': convert_people_cell,
        'Eps':convert_eps_cell
    })
print(df9)

print(df9.to_excel('new.xlsx', sheet_name = 'Stocks'))
print(df9.to_excel('new.xlsx', sheet_name = 'Stocks', startrow = 2, startcol = 2, index = False))

df_stocks = pd.DataFrame({
    'tickers':['GOOGL', 'WMT', 'MSFT'],
    'eps':[845, 65, 64],
    'price':[30.37, 14.26, 30.97],
    'people':[27.82, 4.61, 2.21]
})

df_weather = pd.DataFrame({
    'day':['1/1/2017', '1/5/2017', '1/6/2017'],
    'temp':[23, 34, 252],
    'event':['rainy', 'sunny', 'snow']
})

with pd.ExcelWriter('stocks_weather.xlsx') as writer:
    df_stocks.to_excel(writer, sheet_name='stocks')
    df_weather.to_excel(writer, sheet_name = 'weather')






#Handle Missing Data
import pandas as pd 
df = pd.read_csv('newyork.csv', parse_dates=['Day'])
print(df)
print(type(df.Day[0]))
df.set_index('Day', inplace=True)
print(df)


#Fill Na
x = df.fillna(0)
x = df.fillna({
    'Temp':0,
    'Windspeed':0,
    'Event':'No Event'
})
x = df.fillna(method='ffill')   #Carry forward previous days values.
x = df.fillna(method='bfill')   #Carry forward next days values.
x = df.fillna(method='bfill', axis='columns')  #Axis fill = Horizontal fill
x = df.fillna(method='ffill', limit=1) 
print(x)


#Interpolate 
y = df.interpolate(method='time')   #method = time works only when index set to date time series
print(y)


#Drop Na
z = df.dropna()
z = df.dropna(how='all')      # all column values na in a row
z = df.dropna(thresh=1)       #atleast one column value in a row
z = df.dropna(thresh=2)         #atleast two column value in a row
print(z)


dt = pd.date_range('01/01/2017', '01/11/2017')    # Missing Dates
idx = pd.DatetimeIndex(dt)
print(df.reindex(idx))



# Missing data part 2
import pandas as pd 
import numpy as np

df = pd.read_csv('dates.csv')
print(df)

new_df = df.replace([-99999, -88888],np.NaN)
new_df = df.replace({

    'Temp':-99999,
    'Windspeed':-99999,
    'Event': 'No Event'
},np.NaN)

new_df = df.replace({
    -99999: np.NaN,
    'No Event': 'Sunny'
})
print(new_df)


#REGEX 
new_df = df.replace('[A-Za-z]','',regex=True)
new_df = df.replace({
    'Temp': '[A-Za-z]',
    'Windspeed': '[A-Za-z]'
},'',regex=True)
print(new_df)


#LISTS
import pandas as pd 
import numpy as np
df = pd.DataFrame({
    'score':['exceptional', 'average', 'good', 'poor', 'average', 'exceptional'],
    'student':['rob', 'maya', 'parthiv', 'tom', 'julian', 'erica']
})
print(df)

new_df = df.replace(['poor', 'average', 'good', 'exceptional'], [1, 2, 3, 4])
print(new_df)


#GROUP BY

import pandas as pd 
import matplotlib.pyplot as plt
df = pd.read_csv('city.csv')
g = df.groupby('City')
print(df)
print(g)
for city, city_df in g:
    print(city)
    print(city_df)

print(g.get_group('Mumbai'))  #group by mumbai city 

print(g.max())    #Max of each city
print(g.mean())     #Mean or average per each city

print(g.describe()) 

print(g.plot())
plt.show(g)



#CONCAT DATAFRAMES 
import pandas as pd 
india_weather = pd.DataFrame({
    'city': ['mumbai', 'delhi', 'bangalore'],
    'temp':[23, 34, 28],
    'humidity':[80, 60, 70]
})
print(india_weather)

usa_weather = pd.DataFrame({
    'city': ['NY', 'chicago', 'arizona'],
    'temp':[13, 4, 18],
    'humidity':[85, 50, 75]
})
print(usa_weather)

print(pd.concat([india_weather, usa_weather]))
df = pd.concat([india_weather, usa_weather], ignore_index=True)
print(df)

df = pd.concat([india_weather, usa_weather],  keys = ['india', 'usa'])
print(df)
print(df.loc['usa'])
print(df.loc['india'])

temp_weather = pd.DataFrame({
    'city': ['mumbai', 'delhi', 'bangalore'],
    'temp':[23, 34, 28]
}, index=[0, 1, 2])

humidity_weather = pd.DataFrame({
    'city': [ 'delhi', 'mumbai'],
    'humidity':[60, 80]
}, index=[1, 0])

df = pd.concat([temp_weather, humidity_weather], axis=1)
print(df)

s = pd.Series(['Humid', 'Dry', 'Rain'], name='event')
print(s)

df = pd.concat([temp_weather, s], axis=1)
print(df)



# MERGING IN PANDAS
# Pandas provides a single function, merge, as the entry point for all standard database join operations between DataFrame objects −
# pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None,
# left_index=False, right_index=False, sort=True)
# Here, we have used the following parameters −
# left − A DataFrame object.
# right − Another DataFrame object.
# on − Columns (names) to join on. Must be found in both the left and right DataFrame objects.
# left_on − Columns from the left DataFrame to use as keys. Can either be column names or arrays with length equal to the length of the DataFrame.
# right_on − Columns from the right DataFrame to use as keys. Can either be column names or arrays with length equal to the length of the DataFrame.
# left_index − If True, use the index (row labels) from the left DataFrame as its join key(s). 
# In case of a DataFrame with a MultiIndex (hierarchical), the number of levels must match the number of join keys from the right DataFrame.
# right_index − Same usage as left_index for the right DataFrame.
# how − One of 'left', 'right', 'outer', 'inner'. Defaults to inner. Each method has been described below.
# sort − Sort the result DataFrame by the join keys in lexicographical order.Defaults to True, setting to False will improve the performance substantially in many cases.

import pandas as pd 
df1 = pd.DataFrame({
    'city': ['mumbai', 'delhi', 'bangalore', 'kolkata'],
    'temp':[23, 34, 28, 31],
    'humidity':[80, 60, 65, 50]
})

df2 = pd.DataFrame({
    'city': [ 'delhi', 'mumbai', 'kerala' ],
    'temp':[34, 23, 32],
    'humidity':[60, 80, 45]
})

df = pd.merge(df1, df2, on='city', how = 'inner') #bydefault how = 'inner'
df = pd.merge(df1, df2, on='city', how='outer', indicator=True) # contains all the data
df = pd.merge(df1, df2, on='city', how='left')  # contains all the left join data
df = pd.merge(df1, df2, on='city', how='right')  # contains all the right join data

df = pd.merge(df1, df2, on='city', suffixes=('_left', '_right'))
print(df)



# Pivot Table - Used to summarize and aggregate the data.

import pandas as pd 
df = pd.read_csv('city_all.csv')
print(df)

df1 = df.pivot(index='Day', columns='City', values='Humidity')
df1 = df.pivot(index='Humidity', columns='City')
print(df1)

df2 = df.pivot_table(index='City', columns='Day', margins=True) #default aggfunc = mean
print(df2)

dff = pd.read_csv('weather1.csv')
dff['Day'] = pd.to_datetime(dff['Day'])
print(dff)
print(type(dff['Day'][0]))
df3 = dff.pivot_table(index=pd.Grouper(freq='M', key='Day'), columns='City')
print(df3)


# Reshape DataFrame Using Melt - Transform or Reshape Data.
import pandas as pd 
df = pd.read_csv('Melt.csv')
print(df)

x = pd.melt(df, id_vars=['Day']) 
print(x)
z =  (x[x['variable'] == 'Chicago'])
print(z)

x1 = pd.melt(df, id_vars=['Day'], var_name='City', value_name='Temperature') 
print(x1)
y = (x1[x1['City'] == 'Chicago'])
print(y)



# Stack and Unstack
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
df = pd.read_excel('Stack_Unstack.xlsx', header=[0, 1])
print(df)
print(df.stack())
print(df.stack(level=0))
print(df.unstack())



#Cross Tab - Frequency Distribution

import pandas as pd 
import numpy as np 
df = pd.read_csv('namom.csv')
print(df)
x = pd.crosstab(df.Nationality, df.Handedness)  #shows freq that is number of time occurences.
print(x)
y = pd.crosstab(df.Sex, [df.Handedness, df.Nationality], margins = True)
y = pd.crosstab([df.Nationality, df.Sex], df.Handedness, margins = True)
y = pd.crosstab([df.Nationality], [df.Sex], normalize='index')
y = pd.crosstab([df.Nationality], [df.Sex], margins=True)
y = pd.crosstab([df.Sex], [df.Handedness], values=df.Age, aggfunc=np.average)
print(y)




# Read AND Write Data WITH THE DATABASES
# from sqlalchemy import create_engine

engine = create_engine('postgresql://scott:tiger@localhost:5432/mydatabase')

engine = create_engine('mysql+mysqldb://scott:tiger@localhost/foo')

engine = create_engine('oracle://scott:tiger@127.0.0.1:1521/sidname')

engine = create_engine('mssql+pyodbc://mydsn')

# sqlite://<nohostname>/<path>
# where <path> is relative:
engine = create_engine('sqlite:///foo.db')

# or absolute, starting with a slash:
engine = create_engine('sqlite:////absolute/path/to/foo.db')

import pandas as pd 
import sqlalchemy as sa
urllib is a Python module that can be used for opening URLs. It defines functions and classes to help in URL actions. With Python you can also access and retrieve data from the internet like XML, HTML, JSON, etc. You can also use Python to work with this data directly
import urllib
params = urllib.parse.quote_plus("DRIVER={SQL Server Native Client 11.0};"
                                 "SERVER=LAPTOP-4228MCME\SQLEXPRESS;"
                                 "DATABASE=just;"
                                 "UID=sa;"
                                 "PWD=Password8")

engine = sa.create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))
df = pd.read_sql_table('WORKER71', engine, columns=('FIRST_NAME', 'LAST_NAME', 'SALARY'))
print(df)
query = '''
Select W.FIRST_NAME, W.LAST_NAME, T.WORKER_TITLE
FROM WORKER71 W INNER JOIN TITLE T
ON W.WORKER_ID = T.WORKER_REF_ID
'''
df = pd.read_sql_query(query, engine)
print(df)

x = pd.read_csv('jobs1.csv')

x.rename(columns={
    'workerid':'WORKER_ID',
    'firstname': 'FIRST_NAME',
    'lastname': 'LAST_NAME',
    'salary': 'SALARY',
    'date': 'JOINING_DATE',
    'dept': 'DEPARTMENT'
}, inplace=True)

print(x)

df.to_sql(
    name='WORKER71',
    con=engine,
    index=False,
    if_exists='append'
)

pandas.read_Sql('workers71', engine) #we can also execute query here




#TIMESERIES ANALYSIS = Timeseries is a set of data points indexed in time order.
# M = MONTH; W = WEEK; Q = QUARTER
import pandas as pd 
import matplotlib.pyplot as plt
df = pd.read_csv('appl.csv', parse_dates=['Date'], index_col='Date')
print(df)
print(df.index)
print(df['2017-01'])
print(df['2017-01'].Close.mean())
print(df['2017-01-03'])
print(df['2017-01-07':'2017-01-01'])
print(df['Close'].resample('M').mean())

x = df['Close'].resample('M').mean()
print(x)

x = df['Close'].resample('W').mean()
print(x)


x = df['Close'].resample('Q').mean()
print(x)


 

#TIMESERIES ANALYSIS 
import pandas as pd 
import numpy as np
df = pd.read_csv('appl_no_dates.csv')
print(df.head(5))

rng = pd.date_range(start='6/1/2016', end='6/30/2016', freq='B')
print(rng)

df.set_index(rng, inplace=True)
print(df.head())

print(df['2016-06-01':'2016-06-18'])

print(df['2016-06-01':'2016-06-18'].Close.mean())

print(df.asfreq('D', method='pad'))       #WEEKENDS ALSO INCLUDED

print(df.asfreq('W', method='pad'))       #WEEKLY

print(df.asfreq('H', method='pad'))     #HOURLY

rng = pd.date_range(start='6/1/2016', periods=72, freq='H')
print(rng)
ts = np.random.randint(0, 10, len(rng))
ts = pd.Series(np.random.randint(0, 10, len(rng)), index=rng)
print(ts)




#HOLIDAYS
import pandas as pd 
import numpy as np
from datetime import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.holiday import AbstractHolidayCalendar, nearest_workday, Holiday
from pandas.tseries.offsets import CustomBusinessDay

df = pd.read_csv('appl_no_dates.csv')
print(df)

rng = pd.date_range(start='7/1/2017', end='7/21/2017', freq='B')
print(rng)

us_cal = CustomBusinessDay(calendar=USFederalHolidayCalendar())

rng = pd.date_range(start='7/1/2017', end='7/21/2017', freq=us_cal)
print(rng)

df.set_index(rng, inplace=True)
print(df)

class myCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('My Birth Day', month=4, day=12)    #, observance=nearest_workday),
    ]
    
my_bday = CustomBusinessDay(calendar=myCalendar())
print(pd.date_range('4/1/2017','4/30/2017',freq=my_bday))
print(pd.date_range(start='4/1/2017', end='4/30/2017',freq=my_bday))



egypt_weekdays = "Sun Mon Tue Wed Thu"
b = CustomBusinessDay(weekmask=egypt_weekdays)
print(pd.date_range(start="7/1/2017",periods=20,freq=b))


b = CustomBusinessDay(holidays=['2017-07-04', '2017-07-10'], weekmask=egypt_weekdays)
print(pd.date_range(start="7/1/2017",periods=20,freq=b))

dt = datetime(2017, 7, 9, 0, 0)
print(dt)

print(dt+1*b)



#to_datetime

import pandas as pd
dates = ['2017-01-05', 'Jan 5, 2017', '01/05/2017', '2017.01.05', '2017/01/05','20170105']
print(pd.to_datetime(dates))

dt = ['2017-01-05 2:30:00 PM', 'Jan 5, 2017 14:30:00', '01/05/2016', '2017.01.05', '2017/01/05','20170105']
print(pd.to_datetime(dt))

print(pd.to_datetime('30-12-2016'))

print(pd.to_datetime('5-1-2016'))
print(pd.to_datetime('5-1-2016', dayfirst=True))
print(pd.to_datetime('2017$01$05', format='%Y$%m$%d'))
print(pd.to_datetime('2017#01#05', format='%Y#%m#%d'))

print(pd.to_datetime(['2017-01-05', 'Jan 6, 2017', 'abc'], errors='ignore'))
print(pd.to_datetime(['2017-01-05', 'Jan 6, 2017', 'abc'], errors='coerce'))


current_epoch = 1501324478
print(pd.to_datetime(current_epoch, unit='s'))
print(pd.to_datetime(current_epoch*1000, unit='ms'))

t = pd.to_datetime([current_epoch], unit='s')
print(t)
t.view('int64')
print(t)





# PERIOD AND PERIODINDEX
import pandas as pd 
y = pd.Period('2016')
print(y)
print(y.start_time)
print(y.end_time)
print(y.is_leap_year)

m = pd.Period('2017-12', freq='M')
print(m)
print(m.start_time)
print(m.end_time)
print(m+1)

d = pd.Period('2016-02-28', freq='D')
print(d)
print(d.start_time)
print(d.end_time)
print(d+1)

h = pd.Period('2017-08-15 23:00:00',freq='H')
print(h)
print(h+1)
print(h+pd.offsets.Hour(1))

q1 = pd.Period('2017Q1', freq='Q-JAN')
print(q1)
print(q1.start_time)
print(q1.end_time)
print(q1.asfreq('M',how='start'))
print(q1.asfreq('M',how='end'))

w = pd.Period('2017-07-05',freq='W')
print(w)
print(w-1)
w2 = pd.Period('2017-08-15',freq='W')
print(w2)
print(w2-w)

r = pd.period_range('2011', '2017', freq='q')
print(r)
print(r[0].start_time)
print(r[0].end_time)

r1 = pd.period_range('2011', '2017', freq='q-jan')
print(r1)
print(r1[0].start_time)
print(r1[0].end_time)/

r2 = pd.PeriodIndex(start='2016-01', freq='3M', periods=10)
print(r2)

import numpy as np
ps = pd.Series(np.random.randn(len(r2)), r2)
print(ps)
print(ps.index)
print(ps['2016'])
print(ps['2016':'2017'])
pst = ps.to_timestamp()
print(pst)
print(pst.index)

x = pst.to_period()
print(x)
print(x.index)

df = pd.read_csv('wmt.csv')
print(df)
df.set_index('Line Item', inplace=True)
print(df)
df = df.T
print(df)
print(df.index)


df.index = pd.Period/Index(df.index, freq='Q-JAN')
print(df.index)
print(df.index[0].start_time)
df["Start Date"]=df.index.map(lambda x: x.start_time)
print(df)
df["End Date"]=df.index.map(lambda x: x.end_time)
print(df)




# #TimeZone Handling
# Two types of datetimes in python:-
# 1.Naive (no timezone awareness)
# 2.Timezone aware datetime
import pandas as pd

df = pd.read_csv("ms.csv",index_col='Date' ,parse_dates=['Date'])
print(df)
print(df.index)

df.tz_localize(tz='US/Eastern')
df.index = df.index.tz_localize(tz='US/Eastern')
print(df.index)

df = df.tz_convert('Europe/Berlin')
print(df)
print(df.index)

from pytz import all_timezones
print(all_timezones)

df.index = df.index.tz_convert('Asia/Calcutta')
print(df)

london = pd.date_range('3/6/2012 00:09:00', periods=10, freq='H',tz='Europe/London')
print(london)

td = pd.date_range('3/6/2012 00:00', periods=10, freq='H',tz='dateutil/Europe/London')
print(td)

rng = pd.date_range(start="2017-08-22 09:00:00",periods=10, freq='30min')
print(rng)
s = pd.Series(range(10),index=rng)
print(s)

b = s.tz_localize(tz="Europe/Berlin")
print(b)
print(b.index)

m = s.tz_localize(tz="Asia/Calcutta")
print(m)
print(m.index)

print(b+m)






#SHIFTING AND LAGGING

import pandas as pd
df = pd.read_csv("mj.csv",parse_dates=['Date'],index_col='Date')
print(df)
print(df.shift(1))
print(df.shift(-1))
df['Prev_Day_Price'] = df['Price'].shift(1)
df['Price_Change'] = df['Price'] - df['Prev_Day_Price']
df['5_day_return'] =  (df['Price'] - df['Price'].shift(5))*100/df['Price'].shift(5)
df = df[['Price']]
print(df)
print(df.index)

df.index = pd.date_range(start='2017-08-15',periods=10, freq='B')
print(df)
print(df.index)

print(df.tshift(1))
