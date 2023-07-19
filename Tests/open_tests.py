import pandas as pd

demand = pd.read_csv('20162023_volume.csv')

print(demand.iloc[2]['E1'])