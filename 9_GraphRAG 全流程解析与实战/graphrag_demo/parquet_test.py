import pandas as pd
pd.set_option('display.max_columns', None) # 显示所有列
df = pd.read_parquet('./output/text_units.parquet')
print(df)
