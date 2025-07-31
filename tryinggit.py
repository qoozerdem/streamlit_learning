
import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('output.csv')

freq=[4435.2, 4531.2, 4646.4, 4742.4, 4838.4, 4934.4, 5203.2, 4761.6, 4800.0, 4857.6, 5107.2]

mv_per_mhz = []
valid_freq = []

freqmin= min(freq)
v_min_mean=[]
for i in range(11):
        v_min_mean.append(df.iloc[:, i + 1].mean())
v_min_mean=min(v_min_mean)

print()
print(freqmin)
print((4531.2-freqmin)/(df.iloc[:,  2].mean()-v_min_mean))

for i, f in enumerate(freq):
            if i== 0:
                continue
            if i < df.shape[1] - 1:
                column = df.iloc[:, i + 1]
                avg_mv = column.mean() -v_min_mean
                mv_mhz =  (f - freqmin) /(avg_mv * 1000) 
                mv_per_mhz.append(mv_mhz)
                valid_freq.append(f)
print(mv_per_mhz)