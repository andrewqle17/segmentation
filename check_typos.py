import pandas as pd

data = pd.read_csv("manual_label_sample_combined.csv")


for i in range(len(data)):
    if data.iloc[i, 1].translate({ord(i): None for i in '-_.'})[:-3] != data.iloc[i, 2].translate({ord(i): None for i in '|'}):
        print(data.iloc[i, 0])
