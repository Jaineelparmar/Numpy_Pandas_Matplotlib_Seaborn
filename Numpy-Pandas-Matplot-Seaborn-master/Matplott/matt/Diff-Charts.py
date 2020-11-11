#1. Histogram :
#The histogram represents the frequency of occurrence of specific phenomena which lie within a specific range of values and arranged in consecutive and fixed intervals.

import pandas as pd 
import matplotlib.pyplot as plt 
  
# # create 2D array of table given above 
data = [['E001', 'M', 34, 123, 'Normal', 350], 
        ['E002', 'F', 40, 114, 'Overweight', 450], 
        ['E003', 'F', 37, 135, 'Obesity', 169], 
        ['E004', 'M', 30, 139, 'Underweight', 189], 
        ['E005', 'F', 44, 117, 'Underweight', 183], 
        ['E006', 'M', 36, 121, 'Normal', 80], 
        ['E007', 'M', 32, 133, 'Obesity', 166], 
        ['E008', 'F', 26, 140, 'Normal', 120], 
        ['E009', 'M', 32, 133, 'Normal', 75], 
        ['E010', 'M', 36, 133, 'Underweight', 40] ] 
  
# dataframe created with the above data array 
df = pd.DataFrame(data, columns = ['EMPID', 'Gender',  'Age', 'Sales', 'BMI', 'Income'] ) 
print(df)

# create histogram for numeric data 
df.hist(bins=3) 
  
# show plot 
plt.show() 


#2. Column Chart :
#A column chart is used to show a comparison among different attributes, or it can show a comparison of items over time.

# Plot the bar chart for numeric values 
# a comparison will be shown between 
# all 3 - age, income, sales 

import pandas as pd 
import matplotlib.pyplot as plt 
  
# create 2D array of table given above 
data = [['E001', 'M', 34, 123, 'Normal', 350], 
        ['E002', 'F', 40, 114, 'Overweight', 450], 
        ['E003', 'F', 37, 135, 'Obesity', 169], 
        ['E004', 'M', 30, 139, 'Underweight', 189], 
        ['E005', 'F', 44, 117, 'Underweight', 183], 
        ['E006', 'M', 36, 121, 'Normal', 80], 
        ['E007', 'M', 32, 133, 'Obesity', 166], 
        ['E008', 'F', 26, 140, 'Normal', 120], 
        ['E009', 'M', 32, 133, 'Normal', 75], 
        ['E010', 'M', 36, 133, 'Underweight', 40] ] 
  
# dataframe created with the above data array 
df = pd.DataFrame(data, columns = ['EMPID', 'Gender', 'Age', 'Sales', 'BMI', 'Income'] )
print(df)

df.plot.bar() 
  
# plot between 2 attributes 
plt.bar(df['Age'], df['Sales']) 
plt.xlabel("Age") 
plt.ylabel("Sales") 
plt.show() 




# #3. Box plot chart :
# #A box plot is a graphical representation of statistical data based on the minimum, first quartile, median, third quartile, and maximum. The term “box plot” comes from the fact that the graph looks like a rectangle with lines extending from the top and bottom. Because of the extending lines, this type of graph is sometimes called a box-and-whisker plot. For quantile and median refer to this Quantile and median.

import pandas as pd 
import matplotlib.pyplot as plt 
  
# create 2D array of table given above 
data = [['E001', 'M', 34, 123, 'Normal', 350], 
        ['E002', 'F', 40, 114, 'Overweight', 450], 
        ['E003', 'F', 37, 135, 'Obesity', 169], 
        ['E004', 'M', 30, 139, 'Underweight', 189], 
        ['E005', 'F', 44, 117, 'Underweight', 183], 
        ['E006', 'M', 36, 121, 'Normal', 80], 
        ['E007', 'M', 32, 133, 'Obesity', 166], 
        ['E008', 'F', 26, 140, 'Normal', 120], 
        ['E009', 'M', 32, 133, 'Normal', 75], 
        ['E010', 'M', 36, 133, 'Underweight', 40] ] 
  
# dataframe created with the above data array 
df = pd.DataFrame(data, columns = ['EMPID', 'Gender',  'Age', 'Sales', 'BMI', 'Income'] )
print(df)

# For each numeric attribute of dataframe 
df.plot.box() 
  
# individual attribute box plot 
plt.boxplot(df['Income']) 
plt.show() 



# #4. Pie Chart :
# #A pie chart shows a static number and how categories represent part of a whole the composition of something. A pie chart represents numbers in percentages, and the total sum of all segments needs to equal 100%.

import pandas as pd 
import matplotlib.pyplot as plt 
  
# create 2D array of table given above 
data = [['E001', 'M', 34, 123, 'Normal', 350], 
        ['E002', 'F', 40, 114, 'Overweight', 450], 
        ['E003', 'F', 37, 135, 'Obesity', 169], 
        ['E004', 'M', 30, 139, 'Underweight', 189], 
        ['E005', 'F', 44, 117, 'Underweight', 183], 
        ['E006', 'M', 36, 121, 'Normal', 80], 
        ['E007', 'M', 32, 133, 'Obesity', 166], 
        ['E008', 'F', 26, 140, 'Normal', 120], 
        ['E009', 'M', 32, 133, 'Normal', 75], 
        ['E010', 'M', 36, 133, 'Underweight', 40] ] 
  
# dataframe created with the above data array 
df = pd.DataFrame(data, columns = ['EMPID', 'Gender',  'Age', 'Sales', 'BMI', 'Income'] )
print(df)

plt.pie(df['Age'], labels = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}, autopct ='% 1.1f %%', shadow = True) 
plt.show() 
  
plt.pie(df['Income'], labels = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}, autopct ='% 1.1f %%', shadow = True) 
plt.show() 
  
plt.pie(df['Sales'], labels = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}, autopct ='% 1.1f %%', shadow = True) 
plt.show() 



 
# #5. Scatter plot :
# # A scatter chart shows the relationship between two different variables and it can reveal the distribution trends.
# # It should be used when there are many different data points, and you want to highlight similarities in the data set. 
# # This is useful when looking for outliers and for understanding the distribution of your data.

import pandas as pd 
import matplotlib.pyplot as plt 
  
create 2D array of table given above 
data = [['E001', 'M', 34, 123, 'Normal', 350], 
        ['E002', 'F', 40, 114, 'Overweight', 450], 
        ['E003', 'F', 37, 135, 'Obesity', 169], 
        ['E004', 'M', 30, 139, 'Underweight', 189], 
        ['E005', 'F', 44, 117, 'Underweight', 183], 
        ['E006', 'M', 36, 121, 'Normal', 80], 
        ['E007', 'M', 32, 133, 'Obesity', 166], 
        ['E008', 'F', 26, 140, 'Normal', 120], 
        ['E009', 'M', 32, 133, 'Normal', 75], 
        ['E010', 'M', 36, 133, 'Underweight', 40] ] 
  
dataframe created with 
the above data array 
df = pd.DataFrame(data, columns = ['EMPID', 'Gender', 'Age', 'Sales', 'BMI', 'Income'] )
print(df)

# scatter plot between income and age 
plt.scatter(df['Income'], df['Age']) 
plt.show() 
  
# scatter plot between income and sales 
plt.scatter(df['Income'], df['Sales']) 
plt.show() 
  
# scatter plot between sales and age 
plt.scatter(df['Sales'], df['Age']) 
plt.show() 