import csv
import numpy as np
from PIL import Image
from random import randint


class DataHelper:

    def __init__(self, batch_size, test_idx=None, gray_scale=False, crush=False):
        self.test_idx = test_idx
        self.train_imgs, self.test_imgs = self._get_valid_image_files(test_idx)
        self.train_lbls, self.test_lbls = self._get_labels_no_header(test_idx)
        self.current_idx = 0
        self.batch_size = batch_size
        self._crush = np.vectorize(self._crush)
        self.gray_scale = gray_scale
        self.crush = crush

    def get_next_batch(self):
        end_idx = self.current_idx + self.batch_size

        if end_idx > self.test_idx:
            end_idx = self.test_idx

        data = self._get_image_data(self.train_imgs[self.current_idx:end_idx], augment=False)
        labels = self.train_lbls[self.current_idx:end_idx]

        self.current_idx += self.batch_size

        return (data, labels)

    def get_test_data(self, count=None):
        if count is None:
            data = self._get_image_data(self.test_imgs, augment=False)
            labels = self.test_lbls
            return (data, labels)
        else:
            data = self._get_image_data(self.test_imgs[:count], augment=False)
            labels = self.test_lbls[:count]
            return (data, labels)

    def _get_valid_image_files(self, test_idx=None):
        imgs = []
        with open('./data/valid_images.csv', 'rb') as f:
            reader = csv.reader(f)
            for r in reader:
                for i in r:
                    imgs.append(i)

        # TODO make this an argument
        img_dir = './data/jpeg_redundant_f160/'
        if test_idx is None:
            return [img_dir + i for i in imgs]
        else:
            train_imgs = [img_dir + i for i in imgs[:self.test_idx]]
            test_imgs = [img_dir + i for i in imgs[self.test_idx:]]
            return (train_imgs, test_imgs)

    def _get_labels_no_header(self, test_idx=None):
        labels = []
        with open('./data/valid_image_labels_no_names.csv', 'rb') as f:
            reader = csv.reader(f)
            reader.next()  # Skip header
            for r in reader:
                labels.append(r)

        if test_idx is None:
            return np.array(labels, dtype=np.float32)
        else:
            return (np.array(labels[:test_idx], dtype=np.float32),
                    np.array(labels[test_idx:], dtype=np.float32))

    def _get_image_data(self, image_files, augment=True):
        tmp = []

        for f in image_files:

            img = self._get_image(f, augment)

            arr = np.asarray(img, dtype=np.float32)

            # if we use gray scale pillow drops the channel in shape (45,45)
            # so we have to add it back in
            if self.gray_scale:
                arr = arr.reshape((45, 45, 1))

            tmp.append(arr / 255.0)

        return np.array(tmp)

    def _crush(self, img):
        return 0 if img < 85 else 255

    def _get_image(self, f, augment=True):
        img = Image.open(f)

        if self.gray_scale:
            # convert(L) takes the image to grayscale
            img = img.convert('L')

        mid_x = img.size[0] / 2
        mid_y = img.size[1] / 2

        # crop image to half size, focusing on center
        img = img.crop(
                (
                    mid_x - 113,
                    mid_y - 113,
                    mid_x + 114,
                    mid_y + 114
                )
            )

        if augment:
            img = self._augment_image(img)

        # crush maximizes contrast, only used with gray_scale=True
        if self.crush:
            img = Image.fromarray((self._crush(img)).astype(np.uint8))

        img.thumbnail((45, 45))

        return img

    def _augment_image(self, img):
        # five random pertbuations from D15

        # rotation 0-306
        img = img.rotate(randint(0, 360))

        # shift -4 to 4 for x and y
        img = img.offset(randint(-4, 4), randint(-4, 4))

        # scaling log uniformly sampled between 1.3^-1 and 1.3
        scale = np.exp(np.random.uniform(np.log(1 / 1.3), np.log(1.3)))
        new_size = int(454 * scale)
        img = img.resize((new_size, new_size))

        # flipped probability 0.5
        flip_type = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
        if randint(0, 1):
            img = img.transpose(flip_type[randint(0, 1)])

        # brightness
        # TODO implement, paper says its more expensive

        return img