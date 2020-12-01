import numpy as np
from cv2 import cv2 as cv
import colour
from color_conversion import *
import pickle


class Palette:
    def __init__(self):
        self.L_sampleNum = 40
        self.a_sampleNum = self.b_sampleNum = 40
        total_sampleNum = self.L_sampleNum * self.a_sampleNum * self.b_sampleNum
        LAB_shape = (self.L_sampleNum, self.a_sampleNum, self.b_sampleNum)
        # comparison only happened  between samples in 2 x compareRange
        # LAB[targetL - L_compareIndexRange : targetL + L_compareIndexRange, ..a.., ..b..]
        self.L_compareRange = 30
        self.a_compareRange = self.b_compareRange = 30
        L_compareIndexRange = int(self.L_compareRange * self.L_sampleNum / 100)
        a_compareIndexRange = int(self.a_compareRange * self.a_sampleNum / 255)
        b_compareIndexRange = int(self.b_compareRange * self.b_sampleNum / 255)
        # similar to compare_range, color transfer can only happened between samples in 2 x transferRange
        self.L_transferRange = 12
        self.a_transferRange = self.b_transferRange = 12
        L_transferIndexRange = int(self.L_transferRange * self.L_sampleNum / 100)
        a_transferIndexRange = int(self.a_transferRange * self.a_sampleNum / 255)
        b_transferIndexRange = int(self.b_transferRange * self.b_sampleNum / 255)
        tranfer_totalNum = (2 * L_transferIndexRange) * (2 * a_transferIndexRange) * (2 * b_transferIndexRange)
        assert 2 * L_transferIndexRange <= L_compareIndexRange
        assert 2 * a_transferIndexRange <= a_compareIndexRange
        assert 2 * b_transferIndexRange <= b_compareIndexRange

        L = np.linspace(0, 100, num=self.L_sampleNum)
        a = np.linspace(-128, 127, num=self.a_sampleNum)
        b = np.linspace(-128, 127, num=self.b_sampleNum)
        # [(0 ~ 100), (-128 ~ 127), (-128 ~ 127)]
        standardLAB = np.zeros(shape=(self.L_sampleNum, self.a_sampleNum, self.b_sampleNum, 3))
        standardLAB[:, :, :, 0] = L[:, None, None]
        standardLAB[:, :, :, 1] = a[None, :, None]
        standardLAB[:, :, :, 2] = b[None, None, :]

        ###########################
        standardLAB = standardLAB2visibleLAB(standardLAB)
        ############################

        index_3_dimension = np.linspace(0, total_sampleNum - 1, total_sampleNum, dtype=np.int). \
            reshape((self.L_sampleNum, self.a_sampleNum, self.b_sampleNum))
        nearby_indexes = np.zeros(shape=(self.L_sampleNum, self.a_sampleNum, self.b_sampleNum, tranfer_totalNum),
                                  dtype=np.int)
        # nearby_colors = np.zeros(shape=(self.L_sampleNum, self.a_sampleNum, self.b_sampleNum, tranfer_totalNum, 3))
        nearby_deltaEs = np.zeros(shape=(self.L_sampleNum, self.a_sampleNum, self.b_sampleNum, tranfer_totalNum))
        count = 1
        for L in range(self.L_sampleNum):
            for a in range(self.a_sampleNum):
                for b in range(self.b_sampleNum):
                    count += 1
                    if count % 500 == 0:
                        print("Count : {}".format(count))
                    L_min, a_min, b_min = max(L - L_compareIndexRange, 0), \
                                          max(a - a_compareIndexRange, 0), \
                                          max(b - b_compareIndexRange, 0)
                    L_max, a_max, b_max = min(L + L_compareIndexRange, self.L_sampleNum), \
                                          min(a + a_compareIndexRange, self.a_sampleNum), \
                                          min(b + b_compareIndexRange, self.b_sampleNum)
                    nearby_index = index_3_dimension[L_min: L_max, a_min: a_max, b_min: b_max].reshape(-1)
                    nearby_color = standardLAB[L_min: L_max, a_min: a_max, b_min: b_max].reshape(-1, 3)
                    nearby_deltaE = colour.delta_E(standardLAB[L, a, b], nearby_color)
                    order = np.argsort(nearby_deltaE)
                    nearby_indexes[L, a, b, :tranfer_totalNum] = nearby_index[order][:tranfer_totalNum]
                    # nearby_colors[L, a, b, :tranfer_totalNum] = nearby_color[order][:tranfer_totalNum, :]
                    nearby_deltaEs[L, a, b, :tranfer_totalNum] = nearby_deltaE[order][:tranfer_totalNum]
        self.standardLAB = standardLAB.reshape(-1, 3)
        self.standardLCH = standardLAB2standardLCH(self.standardLAB)
        opencvLAB = standardLAB2opencvLAB(self.standardLAB)
        self.standardRGB = cv.cvtColor(opencvLAB[None, ...], cv.COLOR_LAB2RGB).squeeze() # cause some deviation use int8
        self.nearby_indexes = nearby_indexes.reshape(-1, tranfer_totalNum)
        # self.nearby_colors = nearby_colors.reshape(-1, tranfer_totalNum, 3) # too big to store
        self.nearby_deltaEs = nearby_deltaEs.reshape((-1, tranfer_totalNum))
        self.DP_previous_index = np.zeros(total_sampleNum)  # Can be abandoned
        self.DP_previous_index[:] = np.nan
        self.DP_loss = np.zeros(total_sampleNum)
        self.DP_loss[:] = np.infty

        self.center_point_index = np.where(self.standardLCH[:, 1] < 20)[0]



def create_save_palette():
    palette = Palette()
    file_to_store = open("src/color_model.pkl", "wb")
    pickle.dump(palette, file_to_store)
    file_to_store.close()


def load_palette():
    file_to_load = open("src/color_model.pkl", "rb")    ## ?????
    palette = pickle.load(file_to_load)
    file_to_load.close()
    return palette


if __name__ == "__main__":
    create_save_palette()
    my_palette = load_palette()
    print("oh my god")
