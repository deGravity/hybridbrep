import json
from zipfile import ZipFile
import os
import numpy as np
from PIL import Image

def write_json(obj, filename):
    directory = os.path.dirname(filename)
    os.makedirs(directory, exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(obj, f)

def load_json(p, zf=None):
    if zf is not None:
        if isinstance(zf, str):
            with ZipFile(zf, 'r') as z:
                with z.open(p,'r') as f:
                    return json.load(f)
        if isinstance(zf, ZipFile):
            with zf.open(p,'r') as f:
                return json.load(f)
    with open(p,'r') as f:
        return json.load(f)

class ZippedDataset:
    def __init__(self, root):
        self.index = load_json(root+'.json')
        self.camera = np.load(root+'.npz')
        self.root = root
        self.open_zip()
    
    def close_zip(self):
        self.zip.close()

    def open_zip(self):
        self.zip = ZipFile(self.root+'.zip')

def arr2im(a):
    return Image.fromarray(a.astype(np.uint8))