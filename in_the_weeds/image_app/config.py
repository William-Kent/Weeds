thresholds = dict(
    confidence = 0.5,
    iou = 0.4
)
classes = 12
plot = dict(
    line_thickness = 6
)
weights = dict(
    directory = 'in_the_weeds/weights',
    file_name = 'yolov7_itw.pt'
)
data = dict(
    directory = 'in_the_weeds/data',
    file_name = 'yolov7_itw.yaml'
)
pages = {
    'amaranthus palmeri': 'http://127.0.0.1:8000/amaranthus-palmeri/',
    'ambrosia artemisiifolia': 'http://127.0.0.1:8000/ambrosia-artemisiifolia/',
    'amaranthus tuberculatus': 'http://127.0.0.1:8000/amaranthus-tuberculatus/',
    'eclipta': "http://127.0.0.1:8000/eclipta/",
    'eleusine': 'http://127.0.0.1:8000/eleusine-indica/',
    'euphorbia maculata': 'http://127.0.0.1:8000/euphorbia-maculata/',
    'ipomoea indica': 'http://127.0.0.1:8000/ipomoea-indica/',
    'mollugo verticillata': 'http://127.0.0.1:8000/mollugo-verticillata/',
    'physalis angulata': 'http://127.0.0.1:8000/physalis-angulata/',
    'portulaca oleracea': 'http://127.0.0.1:8000/portulaca-oleracea/',
    'senna obtusifolia': 'http://127.0.0.1:8000/senna-obtusifolia/',
    'sida rhombifolia': 'http://127.0.0.1:8000/sida-rhombifolia/'
}