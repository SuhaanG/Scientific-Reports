import pandas as pd

files = [
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
]

col_presence = {}
for f in files:
    cols = set(c.strip() for c in pd.read_csv(f"data/cicids2018_raw/{f}", nrows=1).columns)
    for c in cols:
        col_presence.setdefault(c, []).append(f)

all_files_count = len(files)
print(f"Total distinct columns across all files: {len(col_presence)}")
print()
print("Columns NOT present in all 10 files:")
for col, present_in in col_presence.items():
    if len(present_in) < all_files_count:
        missing_from = set(files) - set(present_in)
        print(f'  "{col}": present in {len(present_in)}/10 files, MISSING from: {missing_from}')