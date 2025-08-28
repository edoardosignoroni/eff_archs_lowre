import os
import argparse
from datetime import datetime, timedelta

# Argument parser
parser = argparse.ArgumentParser(
    prog='get_train_time',
    description='gets avg time/epoch',
    epilog=''
)

parser.add_argument('-l','--log', action='store',
                    help='log file')

args = parser.parse_args()

log_dir = "/nlp/projekty/mtlowre/new_tokeval/_logs"

log_file = os.path.join(log_dir, args.log)

with open(log_file, "r") as log:
    times = []
    lines = log.readlines()
    for line in lines:
        if "train | epoch" in line:
            line = line.split(' ')
            datel = [int(i) for i in line[0].split("-")]
            timel = [int(i) for i in line[1].split(":")]
            time = datetime(datel[0],datel[1],datel[2],timel[0],timel[1],timel[2])
            times.append(time)
    
epoch_times = [(times[i] - times[i-1]).seconds for i in range(1,len(times))]
print(sum(epoch_times)/len(epoch_times))
