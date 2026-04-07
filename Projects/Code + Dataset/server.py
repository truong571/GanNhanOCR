from flask import Flask, request, render_template
import glob
import os.path as osp
import json
from config import (
    IMAGE_DATASET_ROOT, 
    ORG_IMAGE_DATASET_ROOT,
    HOSTING_ADDR,
    SERVER_HOST,
    SERVER_PORT
)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def lookup():
    if request.method == 'GET':
        return render_template('index.html', lookup_title="Character Lookup")
    if request.form['submit'] == 'character':
        characters = list(filter(lambda x: x != ' ', list(request.form['character'])))
        unicode_values = [hex(ord(character))[2:].upper() for character in characters]
    else:
        unicode_values = request.form['unicode'].split()
        characters = [chr(int(unicode_value, 16)) for unicode_value in unicode_values]
    image_urls = [f'{HOSTING_ADDR}/{IMAGE_DATASET_ROOT}/{unicode_value}.jpg' for unicode_value in unicode_values]
    org_image_urls = [f'{HOSTING_ADDR}/{ORG_IMAGE_DATASET_ROOT}/{unicode_value}.jpg' for unicode_value in unicode_values]
    return render_template(
        'index.html', 
        lookup_title=f"{request.form['submit'].capitalize()} Lookup",
        last_character=request.form.get('character', ''),
        last_unicode=request.form.get('unicode', ''),
        data=zip(characters, unicode_values, image_urls, org_image_urls)
    )

@app.route('/watch', methods=['GET'])
def watch():
    weight_types = ['uniform', 'softmax', 'normalized']
    rep_types = ['norep', 'replace']
    filt_types = ['nofilt', 'filter']
    method_types = ['freq_avgrank', 'avgrank', 'sqravgrank']

    train_folders = glob.glob('archived/trained_new/*')
    train_infos = []
    max_true_rep = 0
    for folder in train_folders:
        train_info = []
        train_info.append(osp.basename(folder).split('_', 1)[1])
        epochs = sorted(list(map(lambda x: osp.basename(x).split('.')[0].split('_'), glob.glob(osp.join(folder, 'epoch_*')))), key=lambda x: (int(x[1]), int(x[3])), reverse=True)
        train_info.append(epochs[0][1])
        train_info.append(epochs[0][3])
        with open(osp.join(folder, 'best_model.json')) as f:
            result = json.load(f)
        train_info.append(result['result']['true_replace'])
        train_info.append(result['result']['false_replace'])
        train_info.append(result['result']['insert'])
        train_info.append(result['result']['delete'])
        train_info.append(result['file'])

        train_infos.append(train_info)

        if result['result']['true_replace'] > max_true_rep:
            max_true_rep = result['result']['true_replace']

    train_infos.sort(key=lambda x: (
        weight_types.index(x[0].rsplit('_',3)[1]),
        rep_types.index(x[0].rsplit('_',3)[3]),
        filt_types.index(x[0].rsplit('_',3)[2]),
        method_types.index(x[0].rsplit('_',3)[0])
    ))

    # for train_info in train_infos:
    #     print(train_info)
    return render_template('watch.html', train_infos=train_infos, max_true_rep=max_true_rep)

if __name__ == '__main__':
    app.run(debug=True, host=SERVER_HOST, port=SERVER_PORT)