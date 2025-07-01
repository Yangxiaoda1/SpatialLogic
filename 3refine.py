import os
image_path='/home/tione/notebook/SpacialLogic-Demo/clips'
for sub1 in os.listdir(image_path):#327
    sub1path=os.path.join(image_path,sub1)
    for sub2 in os.listdir(sub1path):#648642
        sub2path = os.path.join(sub1path, sub2)
        for sub3 in sorted(os.listdir(sub2path)):#Place the held orange in ...
            sub3path= os.path.join(sub2path,sub3)
            if "Retrieve" in sub3:
                continue
            if len(os.listdir(sub3path))>=18:
                idx=0
                for sub4 in sorted(os.listdir(sub3path),reverse=True):#00000.jpg
                    sub4path = os.path.join(sub3path, sub4)
                    print(f'{sub1}/{sub2}/{sub3}/{sub4}')
                    os.remove(sub4path)
                    idx=idx+1
                    if idx>6:
                        break