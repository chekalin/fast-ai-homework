import vgg16

reload(vgg16)
from vgg16 import Vgg16

import math
import csv
import time

total_start_time = time.time()
batch_size = 64
path = 'data/sample/'

vgg = Vgg16()
batches = vgg.get_batches(path + 'train', batch_size=batch_size)
val_batches = vgg.get_batches(path + 'valid', batch_size=batch_size)
vgg.finetune(batches)
vgg.fit(batches, val_batches, nb_epoch=1)

print("starting predictions on test set...")
test_batch_size = 64
test_batches = vgg.get_batches(path + 'test', batch_size=test_batch_size, shuffle=False)

filename_index = 0
file = open('test.csv', "wb")
writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
writer.writerow(["id", "label"])


def get_sample_name():
    return test_batches.filenames[filename_index].replace("unknown/", "").replace(".jpg", "")


prediction_start_time = time.time()
number_of_batches = int(math.ceil(test_batches.n / float(test_batch_size)))
for i in range(number_of_batches):
    batch_start_time = time.time()
    imgs, _ = next(test_batches)
    preds, idxs, classes = vgg.predict(imgs, True)
    for idx in idxs:
        writer.writerow([get_sample_name(), idx])
        filename_index += 1
    print("processing batch {} of {} batches took {} seconds".format(i,
                                                                     number_of_batches,
                                                                     int(time.time() - batch_start_time)))
file.close()
print("Predicting took {} seconds".format(int(time.time() - prediction_start_time)))
print("Total {} seconds".format(int(time.time() - total_start_time)))
