import pandas as pd
from more_itertools import powerset

data = pd.read_csv("combined_segmentation_sample.csv",
                   header=0)
disagreements = pd.read_csv("manual_disagreements_labeled.csv")

# Creates a .csv file whose entries are the rows in which there is at least 1 disagreement
# in cols


def find_disagreements(data, cols):
    filename_pref = ""
    for col in cols:
        filename_pref += col
        filename_pref += "_"
    filename = filename_pref + "disagreements.csv"
    disagreements = data[data[cols].nunique(axis=1) > 1]
    try:
        disagreements.to_csv(filename, mode='x')
    except FileExistsError:
        print("File not saved.\n ", filename, "already exists.")
    return disagreements


cols = ["Manual1", "Manual2", "ChatGPT_4o", "Instant_Segment"]
subsets = powerset(cols)

for subset in subsets:
    if subset == set():
        continue
    find_disagreements(data, list(subset))
