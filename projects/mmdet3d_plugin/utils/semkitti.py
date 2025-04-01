import numpy as np

semantic_kitti_class_frequencies = np.array(
    [
        5.41773033e09,
        1.57835390e07,
        1.25136000e05,
        1.18809000e05,
        6.46799000e05,
        8.21951000e05,
        2.62978000e05,
        2.83696000e05,
        2.04750000e05,
        6.16887030e07,
        4.50296100e06,
        4.48836500e07,
        2.26992300e06,
        5.68402180e07,
        1.57196520e07,
        1.58442623e08,
        2.06162300e06,
        3.69705220e07,
        1.15198800e06,
        3.34146000e05,
    ]
)

kitti_class_names = [
    "empty",
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]

kitti_360_class_frequencies = np.array(
    [
        2264087502,
        20098728,
        104972,
        96297,
        1149426,
        4051087,
        125103,
        105540713,
        16292249,
        45297267,
        14454132,
        110397082,
        6766219,
        295883213,
        50037503,
        1561069,
        406330,
        30516166,
        1950115,
    ]
)


nuscenes_class_frequencies = np.array(
    [
        7777391480, 
        22599798, 
        144076, 
        204134, 
        7304843, 
        8702209, 
        2150955, 
        338454158, 
        75641307, 
        6796153, 
        192733391, 
        191418200, 
        951499
    ]
)