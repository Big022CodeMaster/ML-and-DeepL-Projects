from pathlib import Path

MAINDIR = Path(__file__).parent

def write_file(file, rel_path, json = True):
    file_path = MAINDIR / rel_path
    if json:
        file = json.dumps(file)
    with open(file_path,"w") as output_:
        output_.write(file)