from typing import List
import pandas as pd

def save_data(name: str, data: List[dict]) -> None:
    df = pd.DataFrame(data)
    df.to_csv(f'data/{name}.csv', index=False)