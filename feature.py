from utls import save_url, extract_feature, result_url
import pandas as pd
import numpy as np


class Feature:
    def __init__(self):
        self.data = pd.read_csv(save_url)
        self.feature_label = self.data.apply(extract_feature, axis=1)
        self.feature = []

    def process(self):
        for i in range(0, len(self.feature_label)):
            self.feature.append(np.concatenate((self.feature_label[i][0], self.feature_label[i][1],
                                                self.feature_label[i][2], self.feature_label[i][3],
                                                self.feature_label[i][4]), axis=0))
        data = np.array(self.feature)
        pd.DataFrame(data).to_csv(result_url)


if __name__ == "__main__":
    feature = Feature()
    feature.process()
