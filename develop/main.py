import numpy as np
import pandas as pd

# 通过字典创建DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, np.nan],
        'City': ['New York', 'London', 'Paris']}

df = pd.DataFrame(data)
max_workyear = df['Age'].max()
print(max_workyear)

