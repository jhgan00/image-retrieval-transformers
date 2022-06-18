import os

from .base_dataset import BaseDataset


class InShopDataset(BaseDataset):

    def __init__(self, *args, **kwargs):
        super(InShopDataset, self).__init__(*args, **kwargs)
        assert self.split in {"train", "query", "gallery"}

    def set_paths_and_labels(self):

        with open(os.path.join(self.data_dir, "list_eval_partition.txt")) as f:
            db = f.read().split("\n")[2:-1]

        paths = []
        labels = []
        for line in db:
            line = line.split(" ")
            line = list(filter(lambda x: x, line))
            if line[2] == self.split:
                paths.append(os.path.join(self.data_dir, line[0]))
                labels.append(int(line[1].split("_")[-1]))

        self.paths = paths
        self.labels = labels
