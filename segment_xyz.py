# segment_xyz.py
import pandas as pd
from datetime import datetime
from pathlib import Path

def parse_ts(s):
    s = s.strip()
    try:
        return datetime.strptime(s, "%H:%M:%S.%f")
    except ValueError:
        return datetime.strptime(s, "%H:%M:%S")

infile = Path("data.xyz")

# If data.xyz doesn't exist, create a small synthetic sample (so you can test)
if not infile.exists():
    print("data.xyz not found — creating a small synthetic data.xyz for demo.")
    import numpy as np
    from datetime import timedelta
    start = datetime.strptime("18:45:00.000", "%H:%M:%S.%f")
    rows = []
    for i in range(600):
        t = start + timedelta(seconds=i)
        x = float(i%10) + 0.1  # dummy x
        y = float((i*2)%10) + 0.2
        z = float((i*3)%7) + 0.3
        rows.append([t.strftime("%H:%M:%S.%f")[:-3], x, y, z])
    df = pd.DataFrame(rows)
    df.to_csv(infile, index=False, header=False)
    print("Synthetic data.xyz created.")

raw = pd.read_csv(infile, header=None, names=["timestamp","x","y","z"])
raw['dt'] = raw['timestamp'].apply(parse_ts)

# split timestamps (given by the company)
t1 = parse_ts("18:51:35.232")
t2 = parse_ts("19:02:14.621")

seg1 = raw[raw['dt'] < t1]
seg2 = raw[(raw['dt'] >= t1) & (raw['dt'] <= t2)]
seg3 = raw[raw['dt'] > t2]

seg1.to_csv("segment_before_t1.csv", index=False)
seg2.to_csv("segment_between_t1_t2.csv", index=False)
seg3.to_csv("segment_after_t2.csv", index=False)

print("Wrote: segment_before_t1.csv, segment_between_t1_t2.csv, segment_after_t2.csv")
print(f"Rows: {len(seg1)} {len(seg2)} {len(seg3)}")
