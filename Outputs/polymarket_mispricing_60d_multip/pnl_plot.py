import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('pnl_by_fold.csv')
test = df.loc[df['split']=='test'].pnl_realized_next_step.values
train = df.loc[df['split']=='train'].pnl_realized_next_step.values
val = df.loc[df['split']=='val'].pnl_realized_next_step.values

plt.plot(np.cumsum(test)*-1, label='test')
plt.plot(np.cumsum(train)*-1, label='train')
# plt.plot(np.cumsum(val)*-1, label='val')
plt.legend()
plt.show()
