import csv
import os

from utls import save_url, get_list_dir


class PreprocessorDeepLearning:
    def __init__(self, list_url: str):
        self.list_labels = []
        self.list_url = list_url
        self.tail_str = ".srt"
        self.tail_wav = ".wav"
        self.data = [['url', 'labels']]

    def get_labels(self, dir_name: str):
        dir_name_split = dir_name.split("_")
        if len(dir_name_split) == 2:
            self.list_labels.append(dir_name_split[1])
            return dir_name_split[1]

    def check_is_srt_file(self, name_file: str):
        return name_file.__contains__(self.tail_str)

    def check_is_wav_file(self, name_file: str):
        return name_file.__contains__(self.tail_wav)

    def get_list_detail_url(self, parent_url):
        children_url = []
        for dirname, _, filenames in os.walk(parent_url):
            for file in filenames:
                if self.check_is_wav_file(file):
                    children_url.append(os.path.join(dirname, file))
        return children_url

    def preprocessor(self):
        for url in self.list_url:
            label = self.get_labels(url)

            children_list_url = self.get_list_detail_url(url)
            for child_url in children_list_url:
                data_row = [child_url, label]
                self.data.append(data_row)
        self.save_to_csv(save_url)

    def save_to_csv(self, url):
        file = open(url, 'w+', newline='')
        with file:
            write = csv.writer(file)
            write.writerows(self.data)


if __name__ == "__main__":
    list_dir = get_list_dir()
    preprocessor = PreprocessorDeepLearning(list_dir)
    preprocessor.preprocessor()
