import cv2 as cv
import numpy as np
import os
import random
import sklearn.metrics as sk
import sys


class Image:
    def __init__(self, path: str):
        self.img = self.__get_grayscale_image(path)

        _back_slash_index = path.rindex("\\")
        self.category = int(path[path.rindex("s") + 1:_back_slash_index])
        self.val = int(path[_back_slash_index + 1:path.rindex(".")])

    @staticmethod
    def __get_grayscale_image(path: str):
        return cv.cvtColor(cv.imread(path), cv.COLOR_BGR2GRAY)


class Hist:
    def __init__(self, val: int, cat: int, hist):
        self.hist = hist

        self.val = val
        self.cat = cat


class LBPRrecognation:
    def __init__(self, path: str = 'att_faces'):
        self.__full_img_list = []
        self.__img_histogram_list = []
        self.__img_3x3_histogram_list = []

        self.__path = path

    def __get_all_full_image_inputs(self):
        for subdir, dirs, files in os.walk(self.__path):
            hist_per_cat = []
            for file in files:
                img = Image(os.path.join(subdir, file))
                self.__full_img_list.append(img)
                hist_per_cat.append(Hist(img.val, img.category, self.lbp_hist(img.img)))
            if len(hist_per_cat) > 0:
                self.__img_histogram_list.append(hist_per_cat)

    def __get_all_3x3_image_inputs(self):
        for subdir, dirs, files in os.walk(self.__path):
            hist_per_cat = []
            for file in files:
                img = Image(os.path.join(subdir, file))

                hist_parts_list = []
                for part_img in self.__crop_img(img.img):
                    hist_parts_list.extend(self.lbp_hist(part_img))

                hist_per_cat.append(Hist(img.val, img.category, hist_parts_list))
            if len(hist_per_cat) > 0:
                self.__img_3x3_histogram_list.append(hist_per_cat)

    def initialize_3x3(self):
        self.__get_all_3x3_image_inputs()
        cm, acc, prec = self.__generate_confusion_matrix(self.__img_3x3_histogram_list)
        self.__show_result(cm, acc, prec, 'LBP Image Recognition Using 3x3 Image Segmentation:')

    def initialize(self):
        self.__get_all_full_image_inputs()

        cm, acc, prec = self.__generate_confusion_matrix(self.__img_histogram_list)
        self.__show_result(cm, acc, prec, 'LBP Image Recognition Using Original Image:')

    def __generate_confusion_matrix(self, in_list: list):
        trainings, tests = self.__generate_train_test(in_list)

        predicts = []
        actuals = []
        for test_list in tests:
            for test in test_list:
                test_result = []
                for trainings_list in trainings:
                    for training in trainings_list:
                        test_result.append((self.chi2_distance(test.hist, training.hist), test.cat, training.cat))

                test_result = sorted(test_result, key=lambda x: x[0])
                chi2, predict, actual = test_result[0]
                predicts.append(predict)
                actuals.append(actual)

        np.set_printoptions(threshold=sys.maxsize)

        return sk.confusion_matrix(actuals, predicts), \
               sk.accuracy_score(actuals, predicts) * 100, \
               sk.precision_score(actuals, predicts, average='micro') * 100

    def lbp_hist(self, image):
        temp_array = np.zeros(image.shape, np.uint8)
        row, col = temp_array.shape
        for i in range(row - 1):
            for j in range(col - 1):
                temp_array[i, j] = self.__bin_2_decimal(self.__calc_lbp(image, i, j))

        hist = cv.calcHist([temp_array], [0], None, [256], [0, 256])
        return cv.normalize(hist, hist)

    def __calc_lbp(self, image, i, j):
        sum_lbp = []
        center_pixel = image[i, j]

        sum_lbp.append(self.__lbp_condition(image[i - 1, j], center_pixel))
        sum_lbp.append(self.__lbp_condition(image[i - 1, j + 1], center_pixel))
        sum_lbp.append(self.__lbp_condition(image[i, j + 1], center_pixel))
        sum_lbp.append(self.__lbp_condition(image[i + 1, j + 1], center_pixel))
        sum_lbp.append(self.__lbp_condition(image[i + 1, j], center_pixel))
        sum_lbp.append(self.__lbp_condition(image[i + 1, j - 1], center_pixel))
        sum_lbp.append(self.__lbp_condition(image[i, j - 1], center_pixel))
        sum_lbp.append(self.__lbp_condition(image[i - 1, j - 1], center_pixel))
        return sum_lbp

    @staticmethod
    def chi2_distance(vector_a, vector_b):
        eps = sys.float_info.epsilon
        return 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(vector_a, vector_b)])

    @staticmethod
    def __generate_train_test(_list: list, k: int = .8):
        [random.shuffle(l) for l in _list]

        training = []
        test = []

        for i in range(len(_list)):
            hist = _list[i]
            hists_len = int(len(hist))
            training.append(hist[:int(hists_len * k)])
            test.append(hist[int(hists_len * k):])

        return training, test

    @staticmethod
    def __show_result(cm, acc, prec, title):
        print(20 * '==')
        print(title)
        print('confusion matrix:')
        print(cm)
        print('accuracy score: ', acc)
        print('precision score: ', prec)
        print(20 * '==')



    @staticmethod
    def __crop_img(image, row_size: int = 30, col_size: int = 30):
        windows = []
        row, col = image.shape
        for r in range(0, row - row_size, row_size):
            for c in range(0, col - col_size, col_size):
                windows.append(image[r:r + row_size, c:c + col_size])

        return windows

    @staticmethod
    def __get_grayscale_image(path: str):
        return cv.cvtColor(cv.imread(path), cv.COLOR_BGR2GRAY)

    @staticmethod
    def __lbp_condition(pixel, pixel_c):
        return 1 if pixel > pixel_c else 0

    @staticmethod
    def __bin_2_decimal(binary):
        res = 0
        bit_num = 0
        for i in binary[::-1]:
            res += i << bit_num
            bit_num += 1
        return res


a = LBPRrecognation()
a.initialize()
a.initialize_3x3()




