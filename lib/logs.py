import os
import csv


def write_row(path, data: dict):
    # Recursively create the directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Write the data to the file
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if os.stat(path).st_size == 0:
            writer.writeheader()
        writer.writerow(data)
