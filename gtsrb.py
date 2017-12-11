import cv2
import csv
import os
from .mybase import BaseLoader


class GTSRB(BaseLoader):
    def __init__(self, **kwargs):
        BaseLoader.__init__(self, size=(34, 34), **kwargs)

    def load_data(self):
        train_dir = os.path.join(self.root_dir, 'Final_Training', 'Images')

        for (dir_path, dir_names, files) in os.walk(train_dir):
            dir_name = os.path.basename(dir_path)
            if not dir_name.isdigit():
                continue
            class_id = int(dir_name)
            for file in files:
                if file.endswith('.csv'):
                    continue
                img = cv2.imread(os.path.join(dir_path, file))
                self.x_train.append(img)
                self.y_train.append(class_id)

        test_dir = os.path.join(self.root_dir, 'Final_Test', 'Images')
        gt_file = csv.reader(open(os.path.join(test_dir, 'GT-final_test.csv')), delimiter=';')

        next(gt_file)
        for row in gt_file:
            class_id = int(row[6])
            img = cv2.imread(os.path.join(test_dir, row[0]))
            self.x_test.append(img)
            self.y_test.append(class_id)
