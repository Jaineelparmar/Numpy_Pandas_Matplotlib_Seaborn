import matplotlib.pyplot as plt

# FIRST AND SECOND
x = [1, 2, 3, 4, 5, 6, 7]
y = [50, 51, 52, 53, 54, 55, 47]

plt.xlabel('Day')
plt.ylabel('Temperature')
plt.title('Weather')
plt.plot(x, y, color='green', linewidth=5, linestyle='dashdot')
plt.plot(x, y, 'gD', linewidth=5) #any order works for markers and color
plt.plot(x, y, color='#000480', linewidth=5, linestyle='', marker='+', markersize = 20)
plt.plot(x, y, color='green', alpha=0.5)
plt.show()


# THIRD
days = [1, 2, 3, 4, 5, 6, 7]
max_t = [50, 51, 52, 53, 54, 55, 47]
min_t = [20, 21, 22, 23, 24, 25, 27]
avg_t = [30, 31, 32, 33, 34, 35, 37]

plt.xlabel('Days')
plt.ylabel('Temperature')
plt.title('Weather')

plt.plot(days, max_t, label='Max')
plt.plot(days, min_t, label='Min')
plt.plot(days, avg_t, label='Avg')
plt.legend(loc='best', shadow=True, fontsize='large')
plt.grid()
plt.show()



# BAR PLOT
import numpy as np
company = ['GOOGL', 'AMZN', 'MSFT', 'FB']
revenue = [90, 136, 89, 37]
profit = [40, 2, 34, 12]

plt.xlabel('Company')
plt.ylabel('Revenue')
plt.title('US Tech Stocks')

xpos = np.arange(len(company))
print(xpos)

plt.xticks(xpos, company)
plt.bar(xpos - 0.2, revenue, width=0.4, label='Revenue')
plt.bar(xpos + 0.2, profit, width=0.4, label='Profit')

plt.yticks(xpos, company)
plt.barh(xpos - 0.2, revenue, label='Revenue')
plt.barh(xpos + 0.2, profit, label='Profit')

plt.legend()

plt.bar(company, revenue)
plt.show()




# HISTOGRAMS

blood_sugar = [113, 85, 90, 150, 149, 88, 93, 115, 135, 80, 77, 82, 129]
blood_sugar_men = [113, 85, 90, 150, 149, 88, 93, 115, 135, 80, 77, 82, 129]
blood_sugar_women = [73, 65, 50, 110, 99, 78, 63, 85, 55, 120, 112, 72, 89]

plt.xlabel('Sugar Range')
plt.ylabel('Total Number of Patients')
plt.title('Blood Sugar Analysis')

plt.hist(blood_sugar, bins=3, rwidth=0.95)
print(plt.hist(blood_sugar, bins=[80, 100, 125, 150], rwidth=0.95, color='g'))  #, histtype='step'

plt.hist([blood_sugar_men, blood_sugar_women], bins=[80, 100, 125, 150], rwidth=0.95, color=['green', 'orange'], label=['Men', 'Women'])

plt.hist([blood_sugar_men, blood_sugar_women], bins=[80, 100, 125, 150], rwidth=0.95, color=['green', 'orange'], label=['Men', 'Women'], orientation='horizontal')

plt.legend()
plt.show()





# PIE CHARTS

exp_vals = [1600, 600, 300, 410, 250]
exp_labels = ["Home Rent", "Food", "Phone/Internet Bill", "Car ", "Other Utilities"]

plt.pie(exp_vals, labels = exp_labels, radius = 3, autopct = '%0.1f%%', shadow = True, explode = [0, 0, 1, 0, 1], startangle = 180)
plt.axis('equal')   # Equal aspect ratio ensures that pie is drawn as a circle.

plt.savefig('Figure_6', bbox_inches='tight', pad_inches=2, transparent=True)
# can also save in the pdf

plt.show()


