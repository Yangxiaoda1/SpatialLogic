import os
import json
import math
import shutil
task_path='/home/tione/notebook/SpacialLogic-Demo/task'
for task in os.listdir(task_path):
    taskname = task.split('_')[3].split('.')[0]
    task_desc={}
    task_desc[taskname] = {}
    with open(os.path.join(task_path, task), 'r', encoding='utf-8') as f:
        data = json.load(f)
        for episode in data:
            eid = str(episode['episode_id'])
            task_desc[taskname][eid] = {}
            for pri in episode['label_info']['action_config']:
                for i in range(math.ceil(pri['start_frame']/10), math.floor(pri['end_frame']/10)):
                    task_desc[taskname][eid][i] = pri['action_text']

image_path='/home/tione/notebook/SpacialLogic-Demo/clips'
for sub1 in os.listdir(image_path):#327
    sub1path=os.path.join(image_path,sub1)
    for sub2 in os.listdir(sub1path):#648642
        sub2path = os.path.join(sub1path, sub2)
        for sub3 in os.listdir(sub2path):#00000.jpg
            sub3path= os.path.join(sub2path,sub3)
            if not os.path.isfile(sub3path):
                continue
            idx=int(sub3.split('.')[0])
            if idx in task_desc[sub1][sub2]:
                goaldir=os.path.join(sub2path,task_desc[sub1][sub2][idx])
                os.makedirs(goaldir, exist_ok=True)
                shutil.move(sub3path,goaldir)
            else:
                os.remove(sub3path)