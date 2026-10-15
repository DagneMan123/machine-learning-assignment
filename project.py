#Example of Numpy
import numpy as np

prices = np.array([100.0, 250.0, 50.0, 500.0])
discounted_prices = prices * 0.9

print("Discounted Prices:", discounted_prices)
#example of numpy 2
import numpy as np

flat_array = np.arange(12)

matrix = flat_array.reshape(3, 4)

print("3x4 Matrix:\n", matrix)
#example of numpy 3
import numpy as np

grades = np.array([
    [85, 90, 78],
    [92, 88, 95],
    [75, 80, 85]
])

subject_averages = np.mean(grades, axis=0)
print("Subject Averages:", subject_averages)
#example of numpy 4
import numpy as np

temperatures = np.array([18, 22, 29, 31, 24, 15, 30])

hot_days = temperatures[temperatures > 25]

print("Hot Days (> 25°C):", hot_days)
#example of pandas 1
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'Department': ['HR', 'IT', 'Finance', 'IT']
}

df = pd.DataFrame(data)

print(df)
#example of pandas 2
import pandas as pd

df = pd.DataFrame({
    'Product': ['Laptop', 'Mouse', 'Monitor', 'Keyboard'],
    'Price': [1200, 25, 300, 75],
    'Stock': [10, 150, 5, 45]
})

expensive_items = df[df['Price'] > 100]

print(expensive_items)
#example of pandas 3
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'Department': ['HR', 'IT', 'Finance', 'IT']
}

df = pd.DataFrame(data)

import pandas as pd

df = pd.DataFrame({
    'ID': [1, 2, 3, 4],
    'Salary': [50000, 65000, 80000, 72000]
})

print(df.head(2))

print(df.info())
#example of pandas 4
import pandas as pd

df = pd.DataFrame({
    'Category': ['Electronics', 'Electronics', 'Clothing', 'Clothing', 'Electronics'],
    'Sales': [500, 300, 150, 200, 450]
})

total_sales = df.groupby('Category')['Sales'].sum()

print(total_sales)
#example of Matlab 1
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = {
    'Actual_Spending': [100, 150, 200, 250, 300],
    'Predicted_Spending': [110, 140, 210, 240, 320],
}
df = pd.DataFrame(data)

sns.set_theme(style='whitegrid')

plt.figure(figsize=(8, 5))
sns.regplot(
    x='Actual_Spending',
    y='Predicted_Spending',
    data=df,
    color='purple',
    marker='o',
    scatter_kws={'s': 100},
)

plt.title(
    'ShopSmart: Regression Fit of Actual vs Predicted Spending',
    fontsize=12,
    fontweight='bold',
)
plt.xlabel('Actual Spending ($)')
plt.ylabel('Predicted Spending ($)')
plt.show()
#example of Matlab 2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = {
    'Actual_Spending': [100, 150, 200, 250, 300],
    'Predicted_Spending': [110, 140, 210, 240, 320],
}
df = pd.DataFrame(data)

sns.set_theme(style='darkgrid')
plt.figure(figsize=(8, 5))

sns.kdeplot(
    df['Actual_Spending'], label='Actual Spending', color='blue', fill=True, alpha=0.3
)
sns.kdeplot(
    df['Predicted_Spending'], label='Predicted Spending', color='red', fill=True, alpha=0.3
)

plt.title(
    'ShopSmart: Spending Distribution Comparison',
    fontsize=12,
    fontweight='bold',
)
plt.xlabel('Spending Amount ($)')
plt.legend()
plt.show()
#example of Matlab 3
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = {
    'Customer_ID': [1, 2, 3, 4, 5],
    'Actual_Spending': [100, 150, 200, 250, 300],
    'Predicted_Spending': [110, 140, 210, 240, 320],
}
df = pd.DataFrame(data)

df['Error'] = df['Predicted_Spending'] - df['Actual_Spending']

sns.set_theme(style='whitegrid')
plt.figure(figsize=(8, 5))

sns.barplot(
    x='Customer_ID', y='Error', data=df, palette='coolwarm', hue='Customer_ID', legend=False
)

plt.title(
    'ShopSmart: Prediction Error per Customer ($ y_{pred} - y_{actual} $)',
    fontsize=12,
    fontweight='bold',
)
plt.xlabel('Customer ID')
plt.ylabel('Error Amount ($)')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.show()
#example of Matlab 4
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = {
    'Actual_Spending': [100, 150, 200, 250, 300],
    'Predicted_Spending': [110, 140, 210, 240, 320],
}
df = pd.DataFrame(data)

sns.set_theme(style='ticks')

g = sns.jointplot(
    data=df,
    x='Actual_Spending',
    y='Predicted_Spending',
    kind='reg',
    color='teal',
    height=6,
)

g.fig.suptitle(
    'ShopSmart: Joint Analysis of Actual vs Predicted Spending',
    y=1.03,
    fontsize=12,
    fontweight='bold',
)
g.set_axis_labels('Actual Spending ($)', 'Predicted Spending ($)')
plt.show()
#example of Seaborn 1
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

sns.regplot(x="total_bill", y="tip", data=tips, color="b")

plt.title("Total Bill vs Tip with Regression Line")
plt.show()
#example of Seaborn 2
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

sns.histplot(tips["total_bill"], kde=True, color="purple")

plt.title("Distribution of Total Bills")
plt.xlabel("Total Bill ($)")
plt.show()
#example of Seaborn 3
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

sns.boxplot(x="day", y="total_bill", data=tips, palette="Set2")

plt.title("Total Bill Distribution by Day")
plt.show()
#example of Seaborn 4
import seaborn as sns
import matplotlib.pyplot as plt

flights = sns.load_dataset("flights")
flight_matrix = flights.pivot(index="month", columns="year", values="passengers")

sns.heatmap(flight_matrix, annot=True, fmt="d", cmap="Blues")

plt.title("Passenger Traffic Heatmap")
plt.show()



