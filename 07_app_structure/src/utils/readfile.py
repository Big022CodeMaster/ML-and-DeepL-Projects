import json

def read_file(rel_path, dict = True):
    with open(rel_path,"r") as file:
        output_ = file.read()
    if dict: 
        return json.loads(output_)
    else: 
        return output_
