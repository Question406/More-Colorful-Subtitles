import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
from Palette import *
from color_conversion import *
from functools import reduce
import matplotlib

def _is_in_search_space(hues, hue_range, color_array):
    def _is_in_single_search_space(single_hue, single_hue_range, color_array):
        return np.logical_or(
            np.abs(color_array[:, 2] - single_hue) < single_hue_range,
            np.abs(color_array[:, 2] - (single_hue + 2 * np.pi)) < single_hue_range,
            np.abs(color_array[:, 2] - (single_hue_range - 2 * np.pi)) < single_hue_range
        )

    return reduce(np.logical_or,
                  [_is_in_single_search_space(hue, hue_range, color_array) for hue in hues])

#%% Generate mock data
hue_range = 0.3
my_palette = load_fake_palette()
LCH_background = np.array([[50, 40, 4.0]])
LAB_background = standardLCH2standardLAB(LCH_background)
opposite_hue = oppositeStandardLCH(LCH_background)[:, 2]

search_color_index = np.where(_is_in_search_space(hues=opposite_hue, hue_range=hue_range,
                                                          color_array=my_palette.standardLCH))[0]


RGB_color = my_palette.standardRGB[:] / 255
alpha = np.ones((len(my_palette.standardLAB), 1)) / 20
RGBA_color = np.hstack([RGB_color, alpha])
x = my_palette.standardLAB[:, 1]
y = my_palette.standardLAB[:, 2]
z = my_palette.standardLAB[:, 0]

search_RGB_color = my_palette.standardRGB[search_color_index] / 255
search_alpha = np.ones((len(search_color_index), 1))
search_RGBA_color = np.hstack([search_RGB_color, search_alpha])
search_x = my_palette.standardLAB[search_color_index, 1]
search_y = my_palette.standardLAB[search_color_index, 2]
search_z = my_palette.standardLAB[search_color_index, 0]



# RGBcolor = my_palette.standardRGB[::10]/255
# x = my_palette.standardLAB[::10, 1]
# y = my_palette.standardLAB[::10, 2]
# z = my_palette.standardLAB[::10, 0]

# number_of_datapoints = 30
# x = np.random.rand(number_of_datapoints)
# y = np.random.rand(number_of_datapoints)
# z = np.random.rand(number_of_datapoints)
# color = np.array([0.5,0,0])

fig = plt.figure()
ax3D = fig.add_subplot(111, projection='3d')
ax3D.set_xlim(-128, 128)
ax3D.set_ylim(-128, 128)
ax3D.set_zlim(0, 100)
ax3D.set_xlabel(r'$\mathit{A}$', fontsize=20)
ax3D.set_ylabel(r"$B$", fontsize=20)
ax3D.set_zlabel(r"$L$", fontsize=20)
ax3D.xaxis.set_ticks_position('bottom')
ax3D.spines['bottom'].set_position(('data', 0))
ax3D.scatter(x, y, z, s=10, c=RGBA_color, marker='o')
ax3D.scatter(search_x, search_y, search_z, s=10, c=search_RGBA_color, marker='o')
plt.show()