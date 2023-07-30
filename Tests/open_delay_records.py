import pandas as pd

row = [0, 0, 'actu', 31.689089743589737, 32.89932584269664, 27.911965566714457, 16.17272727272728]

df_delay_records = pd.read_csv('delay_records.csv')
print(list(df_delay_records.columns))
new_row = pd.DataFrame([row], columns=df_delay_records.columns)
print(new_row)
df_delay_records = pd.concat([df_delay_records, new_row], ignore_index=True)

print(df_delay_records)