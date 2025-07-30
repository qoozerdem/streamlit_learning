
import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('output.csv')

freq=[4435.2, 4531.2, 4646.4, 4742.4, 4838.4, 4934.4, 5203.2, 4761.6, 4800.0, 4857.6, 5107.2]



x = df.iloc[:, 10]# Assuming the first column is the x-axis


for i in range(11):
    print(df.iloc[:, i+1].mean())