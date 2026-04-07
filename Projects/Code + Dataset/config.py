DATASET_ROOT = '/home/dtle/nhp/nlp/Project/data'

IMAGE_DATASET_ROOT = f'{DATASET_ROOT}/images'
# IMAGE_DATASET_ROOT = f'{DATASET_ROOT}/org_images'
ORG_IMAGE_DATASET_ROOT = f'{DATASET_ROOT}/org_images'
CHAR_DATASET = f'{DATASET_ROOT}/sinonom_char.txt'
FONT_DATASET_ROOT = f'{DATASET_ROOT}/fonts'

HOSTING_PORT = 3490
HOSTING_HOST = 'localhost'
HOSTING_ADDR = f'http://{HOSTING_HOST}:{HOSTING_PORT}'

SERVER_PORT = 5647
SERVER_HOST = 'localhost'
SERVER_ADDR = f'http://{SERVER_HOST}:{SERVER_PORT}'

FONT_FAMILY = ['NomNaTong-Regular.ttf', 'simsun.ttf', 'simsunb.ttf', 'SimSun-01.ttf', 'PMINGLIU.ttf']
FONT_FAMILY = [f'{FONT_DATASET_ROOT}/{font}' for font in FONT_FAMILY]

# num epoch | % generation | % selections | % cut off

## 1
# schedule = [
#     (20, 1, .7, .5),
#     (10, 1, .5, .4),
#     (5, 1, .5, .3)
# ]

## 2
schedule = [
    (20, 1, .7, .5),
    (10, 1, .5, .5),
    (10, 1, .5, .4),
    (5, 1, .5, .3)
]
# schedule = [
#     (1, 1, .8, .5),
#     (2, 1, .5, .4),
#     (3, 1, .5, .2)
# ]