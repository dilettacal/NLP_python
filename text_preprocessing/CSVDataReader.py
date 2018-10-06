import pandas as pd

class CSVDataReader:

    _TRAIN= "../data/train_plag_data.txt"
    _TEST= "../data/dev_plag_data.txt"

    def _read_train_dataset(self):
        df = pd.read_csv(self._TRAIN, sep=r'\t', header=None, skipinitialspace=True,
                         names=['plag', 'text1', 'text2'], encoding="utf8", engine='python')
        #ParserWarning: Falling back to the 'python' engine because the 'c'
        return df

    def _read_test_dataset(self):
        df = pd.read_csv(self._TEST, sep=r'\t', header=None, skipinitialspace=True,
                         names=['plag', 'text1', 'text2'], encoding="utf8", engine='python')
        #ParserWarning: Falling back to the 'python' engine because the 'c'

        return df

    def get_train(self):
        return self._read_train_dataset()

    def get_test(self):
        return self._read_test_dataset()

