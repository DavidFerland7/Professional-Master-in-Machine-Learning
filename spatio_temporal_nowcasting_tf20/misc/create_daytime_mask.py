import numpy as np
import pandas as pd

df = pd.DataFrame({'time':np.arange(10),'daytime_t=0': np.array([1,1,1,1,1,1,1,0,0,0])})
#df = df.assign({'y':df.x.shift(1)}).fillna({'y': 0})
df = df.assign(
    **{
        't+0': df['daytime_t=0'].shift(0),
        't+1': df['daytime_t=0'].shift(-1),
        't+3': df['daytime_t=0'].shift(-3),
        't+6': df['daytime_t=0'].shift(-6)
    }
).fillna({'t+0':0,'t+1': 0, 't+3': 0, 't+6': 0})


print(df)