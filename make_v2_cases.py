"""Create an exact-time v2 case CSV from the local ERA5 merged file."""
from __future__ import annotations
import argparse, csv
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from netCDF4 import Dataset, num2date
from common import ERA5, CASES

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start', default='2023-07-01T00:00')
    p.add_argument('--count', type=int, default=24)
    p.add_argument('--step-hours', type=int, default=3)
    p.add_argument('--output', type=Path, default=CASES)
    a=p.parse_args()
    start=datetime.fromisoformat(a.start.replace('Z','+00:00')).replace(tzinfo=None)
    with Dataset(ERA5) as ds:
        t=ds.variables['valid_time']; dates=num2date(t[:], units=t.units, calendar=getattr(t,'calendar','standard'), only_use_cftime_datetimes=False)
        by_time={d.replace(tzinfo=None): i for i,d in enumerate(dates)}
    rows=[]
    for i in range(a.count):
        dt=start+timedelta(hours=i*a.step_hours)
        if dt not in by_time: raise ValueError(f'{dt.isoformat()} is not an exact ERA5 valid_time')
        rows.append({'case_id':dt.strftime('V2%Y%m%d_%H%M'),'time':dt.strftime('%Y-%m-%d %H:%M:%S')})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['case_id','time']); w.writeheader(); w.writerows(rows)
    print(f'wrote {len(rows)} exact-time cases to {a.output}')
if __name__=='__main__': main()
