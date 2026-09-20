import pandas as pd
class DataIngestor:
    def load_data(self, path):
        return pd.read_csv(path, index_col='timestamp', parse_dates=True)