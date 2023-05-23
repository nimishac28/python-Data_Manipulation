#!/usr/bin/env python
# coding: utf-8

# # Basic Exercises on Data Importing - Understanding - Manipulating - Analysis - Visualization

# ## Section-1: The pupose of the below exercises (1-7) is to create dictionary and convert into dataframes, how to diplay etc...
# ## The below exercises required to create data 

# ### 1. Import the necessary libraries (pandas, numpy, datetime, re etc)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import re

# set the graphs to show in the jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# set seabor graphs to a better style
sns.set(style="ticks")

### 2. Run the below line of code to create a dictionary and this will be used for below exercises
# In[39]:


raw_data = {"name": ['Bulbasaur', 'Charmander','Squirtle','Caterpie'],
            "evolution": ['Ivysaur','Charmeleon','Wartortle','Metapod'],
            "type": ['grass', 'fire', 'water', 'bug'],
            "hp": [45, 39, 44, 45],
            "pokedex": ['yes', 'no','yes','no']                        
            }


# In[5]:


### 3. Assign it to a object called pokemon and it should be a pandas DataFrame


# In[40]:


pokemon=pd.DataFrame(raw_data)
pokemon


# ### 4. If the DataFrame columns are in alphabetical order, change the order of the columns as name, type, hp, evolution, pokedex

# In[41]:


pokemon= pokemon[['name', 'type','hp','evolution','pokedex']]
pokemon


# ### 5. Add another column called place, and insert places (lakes, parks, hills, forest etc) of your choice.

# In[42]:


pokemon.insert(5,'Places',['lake','park', 'forest','hill'])
pokemon


# ### 6. Display the data type of each column

# In[43]:


pokemon.dtypes


# ### 7. Display the info of dataframe

# In[44]:


pokemon.info


# ## Section-2: The pupose of the below exercise (8-20) is to understand deleting data with pandas.
# ## The below exercises required to use wine.data

# ### 8. Import the dataset *wine.txt* from the folder and assign it to a object called wine
# 
# Please note that the original data text file doesn't contain any header. Please ensure that when you import the data, you should use a suitable argument so as to avoid data getting imported as header.

# In[44]:


wine=pd.read_csv('D:\Learning\\Nimisha\Basic Data Manipulation - Visualization Exercise\wine.txt',header =None)
wine


# ### 9. Delete the first, fourth, seventh, nineth, eleventh, thirteenth and fourteenth columns

# In[45]:


wine.drop(columns=[0,3,6,8,10,12,13],inplace=True,axis=1)


# In[46]:


wine


# ### 10. Assign the columns as below:
# 
# The attributes are (dontated by Riccardo Leardi, riclea '@' anchem.unige.it):  
# 1) alcohol  
# 2) malic_acid  
# 3) alcalinity_of_ash  
# 4) magnesium  
# 5) flavanoids  
# 6) proanthocyanins  
# 7) hue 

# In[47]:


wine.columns=['Alcohol','Malic_acid','Alcalinity_of_ash','Magnesium','Flavanoids','Proanthocyanins','Hue']
wine


# ### 11. Set the values of the first 3 values from alcohol column as NaN

# In[48]:


wine.loc[0:2,'Alcohol':'Alcohol']=np.NaN
wine


# ### 12. Now set the value of the rows 3 and 4 of magnesium as NaN

# In[49]:


wine.loc[3:4,'Magnesium':'Magnesium']=np.NaN
wine


# ### 13. Fill the value of NaN with the number 10 in alcohol and 100 in magnesium

# In[50]:


wine['Alcohol']=wine['Alcohol'].replace(np.NaN,10)
wine['Magnesium']=wine['Magnesium'].replace(np.NaN,100)
wine


# ### 14. Count the number of missing values in all columns.

# In[51]:


wine.isna().sum()


# ### 15.  Create an array of 10 random numbers up until 10 and save it.

# In[52]:


np.random.seed(10)
a=np.random.randint(low=0,high=11,size=10)
a


# ### 16.  Set the rows corresponding to the random numbers to NaN in the column *alcohol*

# In[53]:


def fill_NaN(a, wine):
    for row in a:
        wine.loc[row:row, 'Alcohol':'Alcohol']=np.NaN
fill_NaN(a, wine)
wine.head(11)


# ### 17.  How many missing values do we have now?

# In[54]:


wine.isna().sum()


# ### 18. Print only the non-null values in alcohol

# In[55]:


wine1=wine.loc[wine['Alcohol'].notnull()]
wine1


# ### 19. Delete the rows that contain missing values

# In[56]:


wine=wine.dropna(axis=0)
wine


# ### 20.  Reset the index, so it starts with 0 again

# In[57]:


wine=wine.reset_index(drop=True)
wine


# ## Section-3: The pupose of the below exercise (21-27) is to understand ***filtering & sorting*** data from dataframe.
# ## The below exercises required to use chipotle.tsv

# This time we are going to pull data directly from the internet.  
# Import the dataset directly from this link (https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv) and create dataframe called chipo

# In[222]:


import requests
url="https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv"
r=requests.get(url)
c=r.text

with open("my_dummy_file.txt", "w") as file:
    file.write(c)

chipo = pd.read_csv("my_dummy_file.txt", sep="\t")
chipo


# ### 21. How many products cost more than $10.00? 
# 
# Use `str` attribute to remove the $ sign and convert the column to proper numeric type data before filtering.
# 

# In[223]:


chipo['item_price']=chipo['item_price'].str.replace('$','').astype('float')
chipo[chipo['item_price']>10]


# ### 22. Print the Chipo Dataframe & info about data frame

# In[224]:


print(chipo)
print(chipo.info())


# ### 23. What is the price of each item? 
# - Delete the duplicates in item_name and quantity
# - Print a data frame with only two columns `item_name` and `item_price`
# - Sort the values from the most to less expensive

# In[228]:


chipo.drop_duplicates(subset=['item_name','quantity'], inplace=True)
chipo[['item_name','item_price']].sort_values(by=['item_price'],ascending=False)


# ### 24. Sort by the name of the item

# In[229]:


chipo.sort_values(by=['item_name'])


# ### 25. What was the quantity of the most expensive item ordered?

# In[234]:


chipo.sort_values(by=['item_price'],ascending=False).head(1)


# ### 26. How many times were a Veggie Salad Bowl ordered?

# In[252]:


count=chipo.query("item_name=='Veggie Salad Bowl'")
count


# ### 27. How many times people orderd more than one Canned Soda?

# In[269]:


count1=chipo.query("item_name=='Canned Soda'")
count1[count1['quantity'].apply(lambda x: x>1)]


# ## Section-4: The purpose of the below exercises is to understand how to perform aggregations of data frame
# ## The below exercises (28-33) required to use occupation.csv

# ###  28. Import the dataset occupation.csv and assign object as users

# In[214]:


users=pd.read_csv('D:\Learning\\Nimisha\Basic Data Manipulation - Visualization Exercise\occupation.csv')
users = users['user_id|age|gender|occupation|zip_code'].str.split('|', expand=True)
users.columns =['user_id', 'age', 'gender', 'occupation', 'zip_code']
users


# ### 29. Discover what is the mean age per occupation

# In[215]:


users.age=users.age.astype('int')
users.groupby('occupation').age.mean()


# ### 30. Discover the Male ratio per occupation and sort it from the most to the least.
# 
# Use numpy.where() to encode gender column.

# In[216]:


users['M']=np.where(users.gender=='M',True, False)
Male_ratio= (users.groupby('occupation').M.sum()/users.groupby('occupation').gender.count()).sort_values(ascending=False)
Male_ratio


# ### 31. For each occupation, calculate the minimum and maximum ages

# In[217]:


users.groupby('occupation').agg({'age': ['min', 'max']})


# ### 32. For each combination of occupation and gender, calculate the mean age

# In[218]:


users.groupby(['occupation','gender']).age.mean()


# ### 33.  For each occupation present the percentage of women and men

# In[220]:


users['F']=np.where(users.gender=='F',True, False)
Female_ratio=(users.groupby('occupation').F.sum()/users.groupby('occupation').gender.count()*100)
Male_ratio=(users.groupby('occupation').M.sum()/users.groupby('occupation').gender.count()*100)
df=pd.concat([ Female_ratio, Male_ratio],axis=1)
df.columns=  ['Female','Male']
df


# ## Section-6: The purpose of the below exercises is to understand how to use lambda-apply-functions
# ## The below exercises (34-41) required to use student-mat.csv and student-por.csv files 

# ### 34. Import the datasets *student-mat* and *student-por* and append them and assigned object as df

# In[87]:


student_mat=pd.read_csv('D:\Learning\\Nimisha\Basic Data Manipulation - Visualization Exercise\student-mat.csv')
student_por=pd.read_csv('D:\Learning\\Nimisha\Basic Data Manipulation - Visualization Exercise\student-por.csv')

df=student_mat.append(student_por, ignore_index=True)
df


# ### 35. For the purpose of this exercise slice the dataframe from 'school' until the 'guardian' column

# In[88]:


df=df.loc[:,'school':'guardian']
df


# ### 36. Create a lambda function that captalize strings (example: if we give at_home as input function and should give At_home as output.

# In[89]:


def capital():
    return lambda a:a.upper()


# ### 37. Capitalize both Mjob and Fjob variables using above lamdba function

# In[90]:


print(df.Mjob.apply(capital()))
print(df.Fjob.apply(capital()))


# ### 38. Print the last elements of the data set. (Last few records)

# In[91]:


df.tail()


# ### 39. Did you notice the original dataframe is still lowercase? Why is that? Fix it and captalize Mjob and Fjob.

# In[92]:


df['Mjob']=df.Mjob.apply(capital())
df['Fjob']=df.Fjob.apply(capital())
df


#  ### 40. Create a function called majority that return a boolean value to a new column called legal_drinker

# In[93]:


def majority(df):
    df['Legal_drinker'] =df.age.apply(lambda a: True if a>=18  else False)
    return df
x=majority(df)
x


# ### 41. Multiply every number of the dataset by 10.

# In[97]:


df.select_dtypes(include=['int64']).apply(lambda a: a*10)


# ## Section-6: The purpose of the below exercises is to understand how to perform simple joins
# ## The below exercises (42-48) required to use cars1.csv and cars2.csv files 

# ### 42. Import the datasets cars1.csv and cars2.csv and assign names as cars1 and cars2

# In[9]:


cars1=pd.read_csv('D:\Learning\\Nimisha\Basic Data Manipulation - Visualization Exercise\cars1.csv')
cars2=pd.read_csv('D:\Learning\\Nimisha\Basic Data Manipulation - Visualization Exercise\cars2.csv')


#    ### 43. Print the information to cars1 by applying below functions 
#    hint: Use different functions/methods like type(), head(), tail(), columns(), info(), dtypes(), index(), shape(), count(), size(), ndim(), axes(), describe(), memory_usage(), sort_values(), value_counts()
#    Also create profile report using pandas_profiling.Profile_Report

# In[10]:


type(cars1)
cars1.head()
cars1.tail()
cars1.columns
cars1.info()
cars1.dtypes
cars1.index
cars1.shape
cars1.count()
cars1.size
cars1.ndim
cars1.axes
cars1.describe()
cars1.memory_usage()
cars1.sort_values
cars1.value_counts

from pandas_profiling import ProfileReport


# In[232]:


Profile_Report=ProfileReport(cars1)
Profile_Report


# ### 44. It seems our first dataset has some unnamed blank columns, fix cars1

# In[157]:


cars1.dropna(axis=1,inplace=True)
cars1


# ### 45. What is the number of observations in each dataset?

# In[148]:


print(len(cars1))
print(len(cars2))


# ### 46. Join cars1 and cars2 into a single DataFrame called cars

# In[159]:


cars=cars1.append(cars2, ignore_index=True)
cars


# ### 47. There is a column missing, called owners. Create a random number Series from 15,000 to 73,000.

# In[165]:


x=pd.Series(np.random.randint(15000,73000,398))
x


# ### 48. Add the column owners to cars

# In[167]:


cars['owners']=x
cars


# ## Section-7: The purpose of the below exercises is to understand how to perform date time operations

# ### 49. Write a Python script to display the
# - a. Current date and time
# - b. Current year
# - c. Month of year
# - d. Week number of the year
# - e. Weekday of the week
# - f. Day of year
# - g. Day of the month
# - h. Day of week

# In[195]:


sysdate=pd.Timestamp.now()
sysdate


# In[187]:


sysdate.year


# In[188]:


sysdate.month


# In[191]:


sysdate.week


# In[210]:


sysdate.day_name()


# In[201]:


sysdate.day_of_year


# In[203]:


sysdate.day


# In[209]:


sysdate.dayofweek


# ### 50. Write a Python program to convert a string to datetime.
# Sample String : Jul 1 2014 2:43PM 
# 
# Expected Output : 2014-07-01 14:43:00

# In[229]:


date=input('Enter the date and time(mon-dd-yyyy h:m am/pm):')
try:
    date1=dt.datetime.strptime(date,'%b %d %Y %I:%M%p')
    date1.strftime('%Y-%m-%d %H:%M:%S')
except Exception as e:
    print('Input date should be in given format')


# ### 51. Write a Python program to subtract five days from current date.
# 
# Current Date : 2015-06-22
# 
# 5 days before Current Date : 2015-06-17

# In[237]:


date2='2015-06-22'
date2=dt.datetime.strptime(date2,'%Y-%m-%d')
days=dt.timedelta(5)
new_date=date2-days
new_date.strftime('%Y-%m-%d')


# ### 52. Write a Python program to convert unix timestamp string to readable date.
# 
# Sample Unix timestamp string : 1284105682
#     
# Expected Output : 2010-09-10 13:31:22

# In[6]:


epoch=int(input('Enter the unix series:'))
date=dt.datetime.fromtimestamp(epoch)
date.strftime('%Y-%m-%d %H:%M:%S')


# ### 53. Convert the below Series to pandas datetime : 
# 
# DoB = pd.Series(["07Sep59","01Jan55","15Dec47","11Jul42"])
# 
# Make sure that the year is 19XX not 20XX

# In[20]:


DoB = pd.Series(["07Sep59","01Jan55","15Dec47","11Jul42"])
date=pd.to_datetime(DoB)-pd.DateOffset(years=100)
date


# ### 54. Write a Python program to get days between two dates. 

# In[28]:


date_a=pd.to_datetime('27-11-1986')
date_b=pd.to_datetime('28-12-1992')
days=date_b-date_a
days


# ### 55. Convert the below date to datetime and then change its display format using the .dt module
# 
# Date = "15Dec1989"
# 
# Result : "Friday, 15 Dec 98"

# In[33]:


date="15Dec1989"
a=dt.datetime.strptime(date,'%d%b%Y')
c=dt.datetime.strftime(a,'%A, %d %b %y')
c


# ## The below exercises (56-66) required to use wind.data file 

# ### About wind.data:
# 
# The data have been modified to contain some missing values, identified by NaN.  
# 
# 1. The data in 'wind.data' has the following format:
"""
Yr Mo Dy   RPT   VAL   ROS   KIL   SHA   BIR   DUB   CLA   MUL   CLO   BEL   MAL
61  1  1 15.04 14.96 13.17  9.29   NaN  9.87 13.67 10.25 10.83 12.58 18.50 15.04
61  1  2 14.71   NaN 10.83  6.50 12.62  7.67 11.50 10.04  9.79  9.67 17.54 13.83
61  1  3 18.50 16.88 12.33 10.13 11.17  6.17 11.25   NaN  8.50  7.67 12.75 12.71
"""
The first three columns are year, month and day.  The remaining 12 columns are average windspeeds in knots at 12 locations in Ireland on that day. 
# ### 56. Import the dataset wind.data and assign it to a variable called data and replace the first 3 columns by a proper date time index

# In[83]:


data=pd.read_csv('D:\Learning\\Nimisha\Basic Data Manipulation - Visualization Exercise\wind.data',parse_dates=[['Yr', 'Mo','Dy']])
data


# ### 57. Year 2061 is seemingly imporoper. Convert every year which are < 70 to 19XX instead of 20XX.

# In[84]:


data.rename(columns={'Yr_Mo_Dy':'Date'}, inplace=True)
    


# In[85]:


data['Date']=np.where(data['Date'].astype('str').str[0:4].astype('int')>2000, data['Date']-pd.DateOffset(years=100), data['Date'])
data


# ### 58. Set the right dates as the index. Pay attention at the data type, it should be datetime64[ns].

# In[86]:


data=data.set_index('Date')
data.index.astype('datetime64[ns]')
data


# ### 59. Compute how many values are missing for each location over the entire record.  
# #### They should be ignored in all calculations below. 

# In[88]:


data.isna().sum()


# ### 60. Compute how many non-missing values there are in total.

# In[89]:


(data.isna()==False).sum().sum()


# ### 61. Calculate the mean windspeeds over all the locations and all the times.
# #### A single number for the entire dataset.

# In[90]:


x=data.mean().mean()
x


# ### 62. Create a DataFrame called loc_stats and calculate the min, max and mean windspeeds and standard deviations of the windspeeds at each location over all the days 
# 
# #### A different set of numbers for each location.

# In[91]:


loc_stats=data.describe().loc[['min','max','mean','std'],:]
loc_stats


# ### 63. Create a DataFrame called day_stats and calculate the min, max and mean windspeed and standard deviations of the windspeeds across all the locations at each day.
# 
# #### A different set of numbers for each day.

# In[92]:


data.T.describe().loc[['min','max','mean','std'],:]


# ### 64. Find the average windspeed in January for each location.  
# #### Treat January 1961 and January 1962 both as January.

# In[93]:


data[(data.index.month==1)].mean()


# ### 65. Calculate the mean windspeed for each month in the dataset.  
# #### Treat January 1961 and January 1962 as *different* months.
# #### (hint: first find a  way to create an identifier unique for each month.)

# In[94]:


data.groupby([data.index.year,data.index.month]).mean()


# ### 66. Calculate the min, max and mean windspeeds and standard deviations of the windspeeds across all locations for each week (assume that the first week starts on January 2 1961) for the first 52 weeks.

# In[131]:


start_date=data.index[0]+pd.DateOffset(days=1)
end_date=start_date+pd.DateOffset(weeks=52)
filtered_df= data[(data.index >= start_date) & (data.index <= end_date)].describe().loc[['min','max','mean','std']]
filtered_df


# ## The below exercises (67-70) required to use appl_1980_2014.csv  file

# ### 67. Import the file appl_1980_2014.csv and assign it to a variable called 'apple'

# In[3]:


apple=pd.read_csv('D:\Learning\\appl_1980_2014.csv')
apple


# ### 68.  Check out the type of the columns

# In[4]:


apple.info()


# ### 69. Transform the Date column as a datetime type

# In[5]:


apple.Date=apple.Date.astype('datetime64[ns]')


# ### 70.  Set the date as the index

# In[6]:


apple['Date'] = pd.to_datetime(apple['Date'])
apple=apple.set_index('Date')
apple


# ### 71.  Is there any duplicate dates?

# In[7]:


result= 'No' if (apple.index.duplicated().any()== False) else 'Yes'
result


# ### 72.  The index is from the most recent date. Sort the data so that the first entry is the oldest date.

# In[8]:


apple=apple.sort_index()
apple


# ### 73. Get the last business day of each month

# In[9]:


apple_reset = apple.reset_index()
# apple_reset
apple_reset['year'] = apple_reset.Date.dt.strftime('%Y')
apple_reset['month'] = apple_reset.Date.dt.strftime('%m')
lbd = apple_reset.pivot_table(values='Date', index=['year','month'], aggfunc=max)
lbd


# ### 74.  What is the difference in days between the first day and the oldest

# In[11]:


first_day=apple.index.min()
oldest_day=apple.index.max()

date_diff= oldest_day-first_day
date_diff


# ### 75.  How many months in the data we have?

# In[12]:


unique_months=apple.index.month.unique()
len(unique_months)


# ## Section-8: The purpose of the below exercises is to understand how to create basic graphs

# ### 76. Plot the 'Adj Close' value. Set the size of the figure to 13.5 x 9 inches

# In[27]:


apple.plot(y='Adj Close',figsize=(13.5,9))


# ## The below exercises (77-80) required to use Online_Retail.csv file

# ### 77. Import the dataset from this Online_Retail.csv and assign it to a variable called online_rt

# In[55]:


online_rt=pd.read_csv("D:\\personal\\nimisha\\Online_Retail.csv", encoding='windows-1252')
online_rt


# ### 78. Create a barchart with the 10 countries that have the most 'Quantity' ordered except UK

# In[57]:


online_rt.groupby('Country').Quantity.sum().sort_values(ascending =False).drop('United Kingdom').head(10).plot(kind='bar')


# ### 79.  Exclude negative Quatity entries

# In[58]:


online_rt= online_rt[online_rt['Quantity'] >= 0]
online_rt


# ### 80. Create a scatterplot with the Quantity per UnitPrice by CustomerID for the top 3 Countries
# Hint: First we need to find top-3 countries based on revenue, then create scater plot between Quantity and Unitprice for each country separately
# 

# In[60]:


sns.set(rc={'figure.figsize': (12, 8)})
customers = online_rt.groupby(['CustomerID','Country']).sum()
customers = customers[customers.UnitPrice > 0]
customers['Country'] = customers.index.get_level_values(1) #getting value of index(1) in the column country
top_countries =  ['United Kingdom', 'Netherlands', 'EIRE']
customers = customers[customers['Country'].isin(top_countries)]
g = sns.FacetGrid(customers, col="Country")
g.map(plt.scatter, "Quantity", "UnitPrice", alpha=1)


# ## The below exercises (81-90) required to use FMCG_Company_Data_2019.csv file

# ### 81. Import the dataset FMCG_Company_Data_2019.csv and assign it to a variable called company_data

# In[2]:


company_data=pd.read_csv('D:\Learning\\Nimisha\Basic Data Manipulation - Visualization Exercise\\FMCG_Company_Data_2019.csv')
company_data


# ### 82. Create line chart for Total Revenue of all months with following properties
# - X label name = Month
# - Y label name = Total Revenue

# In[3]:


sns.set(rc={'figure.figsize': (10, 6)})
sns.lineplot(x='Month',y='Total_Revenue', data=company_data)

plt.xlabel('Month')
plt.ylabel('Total Revenue')
plt.title('Total Revenue of All Months')
plt.show()


# ### 83. Create line chart for Total Units of all months with following properties
# - X label name = Month
# - Y label name = Total Units
# - Line Style dotted and Line-color should be red
# - Show legend at the lower right location.

# In[4]:


sns.set(rc={'figure.figsize': (10, 6)})
sns.lineplot(x=company_data.Month,y=company_data.Total_Units, linestyle=':', linewidth=1.5, color='red')
plt.legend(['Total_Units'],loc='lower right')
plt.show()


# ### 84. Read all product sales data (Facecream, FaceWash, Toothpaste, Soap, Shampo, Moisturizer) and show it  using a multiline plot
# - Display the number of units sold per month for each product using multiline plots. (i.e., Separate Plotline for each product ).

# In[5]:


sns.set(rc={'figure.figsize': (10, 6)})
sns.lineplot(x=company_data.Month,y=company_data.FaceCream, linestyle='-', label="Face cream Sales Data", marker='o' )
sns.lineplot(x=company_data.Month,y=company_data.FaceWash, linestyle='-', label="Face Wash Sales Data", marker='o' )
sns.lineplot(x=company_data.Month,y=company_data.ToothPaste, linestyle='-', label="Toothpaste Sales Data", marker='o' )
sns.lineplot(x=company_data.Month,y=company_data.Soap, linestyle='-', label="Soap Sales Data", marker='o' )
sns.lineplot(x=company_data.Month,y=company_data.Shampo, linestyle='-', label="Shampoo Sales Data", marker='o' )
sns.lineplot(x=company_data.Month,y=company_data.Moisturizer, linestyle='-', label="Moisturizer Sales Data", marker='o' )
plt.xlabel("Months")
plt.ylabel("Sales units in number")
plt.legend(loc=2)
plt.show()


# ### 85. Create Bar Chart for soap of all months and Save the chart in folder

# In[6]:


sns.barplot(x=company_data.Month, y=company_data.Soap,color='Yellow')
plt.xlabel('Month')
plt.ylabel('Soap Quantity')
folder_path='D:\\Learning'
file_name='Soap_chart.png'
plt.savefig(folder_path + file_name)
plt.show()


# ### 86. Create Stacked Bar Chart for Soap, Shampo, ToothPaste for each month
# The bar chart should display the number of units sold per month for each product. Add a separate bar for each product in the same chart.

# In[14]:


sum1=company_data.pivot_table(index='Month',values=['ToothPaste','Soap','Shampo'])
sum1.plot(kind='bar',stacked=True)


# ### 87. Create Histogram for Total Revenue

# In[19]:


sns.displot(company_data.Total_Revenue,kind='hist')


# ### 88. Calculate total sales data (quantity) for 2019 for each product and show it using a Pie chart. Understand percentage contribution from each product

# In[32]:


sum2=company_data.loc[ :,['FaceCream','FaceWash','ToothPaste','Soap','Shampo','Moisturizer']].sum()
sum2.plot(kind='pie',label='Sale')


# ### 89. Create line plots for Soap & Facewash of all months in a single plot using Subplot

# In[46]:


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

axes[0].plot(company_data['Month'], company_data['Soap'], marker='o', linestyle='-', color='blue')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Soap')
axes[0].set_title('Soap Sales')

axes[1].plot(company_data['Month'], company_data['FaceWash'], marker='o', linestyle='-', color='green')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Facewash')
axes[1].set_title('Facewash Sales')

plt.tight_layout()


# In[ ]:





# ### 90. Create Box Plot for Total Profit variable

# In[35]:


company_data.Total_Profit.plot(kind='box')


# In[ ]:




