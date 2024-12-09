import os

from pipelines.dependencies.dataset_savers.dataset_saver import DatasetSaver
from uuid import uuid4


class YoloDatasetSaver(DatasetSaver):
    def __init__(self, boat_category : int):
        super().__init__()
        self.boat_category = boat_category

    def save(self, path: str):
        if path in os.listdir(): raise FileExistsError()
        os.makedirs(path, exist_ok=True)
        os.makedirs(f'{path}/images')
        os.makedirs(f'{path}/images/train')
        os.makedirs(f'{path}/images/val')
        os.makedirs(f'{path}/labels')
        os.makedirs(f'{path}/labels/train')
        os.makedirs(f'{path}/labels/val')
        for current in range(0, len(self.train_images)):
            id = str(uuid4())
            img = self.train_images[current]
            img.save(f'{path}/images/train/{id}.jpg')
            labels = self.get_train_labels_from_image(img)
            self.create_label_file(id, labels, path)
        dataset_yaml = open(f'{path}/qaisc.yaml', "x")
        dataset_yaml.write(
            f"""# This dataset has been generated by QAISC synthetic dataset generator.
path: ./
train: images/train
val: images/val
test:

names:
    {self.boat_category}: boat
            """
        )
        dataset_yaml.close()

    def create_label_file(self, id, labels, path):
        text = ''
        for x in labels:
            text += f'{self.boat_category} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f} {x[3]:.6f}\n'
        label_file = open(f'{path}/labels/train/{id}.txt', "x")
        label_file.write(text)
        label_file.close()







