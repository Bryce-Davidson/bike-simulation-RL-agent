import os
import csv


def write_row(path, data: dict):
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if os.stat(path).st_size == 0:
            writer.writeheader()
        writer.writerow(data)
