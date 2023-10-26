from flask import Flask, request, render_template
import random
import os
import json
import itertools
from datetime import datetime

app = Flask(__name__)

def random_images():
    chosen_folders = random.sample(folders, 2)
    url = []
    for folder in chosen_folders:
        scene = {}
        scene['id'] = folder
        scene['whole'] = os.path.join(relative_path,folder,'whole.png')
        scene['Rooms'] = {}
        rooms = [dir for dir in os.listdir(os.path.join(base,folder)) if os.path.isdir(os.path.join(base,folder,dir))]
        
        for i, room in enumerate(rooms):
            scene['Rooms'][f'Room {i + 1}'] = {}
            scene['Rooms'][f'Room {i + 1}']['Top View'] = os.path.join(relative_path,folder,room,'top.png')
            scene['Rooms'][f'Room {i + 1}']['CCTV View'] = os.path.join(relative_path,folder,room,'cctv.png')
            scene['Rooms'][f'Room {i + 1}']['Closeup View'] = os.path.join(relative_path,folder,room,'closeup.png')
        url.append(scene)
    return url

@app.route("/")
@app.route("/render_page") 
def hello():
    url = random_images()
    return render_template('index.html', url = url)

@app.route("/get_img")
def get_img():
    urls = random_images()
    return urls

@app.route('/receive_choice',methods=["GET", "POST"])
def receive_choice():
    if request.method=='POST':
        now = datetime.now()
        result[now.strftime("%y%m%d_%H%H%S")] = (request.form.get('better'), request.form.get('worse'))
        with open('./results/result.json', 'w') as f:
            json.dump(result, f)
    return 'nothing'


if __name__ == '__main__':
    base = './static/imgs'
    relative_path = '/imgs'
    folders = [folder for folder in os.listdir(base) if not folder.startswith('.') ]

    result = dict()

    app.run(port=8001, debug=True)