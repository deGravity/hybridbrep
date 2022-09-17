import torch_geometric as tg
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
import random
import torch
import json
from tqdm import tqdm
from torch_geometric.nn import global_max_pool
from torch.nn import Linear
from torch.nn.functional import leaky_relu, cross_entropy
from collections import Counter
from math import ceil

def create_subset(data, seed, size):
    random.seed(seed)
    return random.sample(data, size)

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = tg.nn.GCNConv(in_channels, 2*out_channels)
        self.conv2 = tg.nn.GCNConv(2*out_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class CodeGraphAutoencoder(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = tg.nn.GAE(GCNEncoder(64,16))
    def forward(self, batch):
        return self.model.encode(batch.x, batch.edge_index)
    def training_step(self, batch, batch_idx):
        z = self(batch)
        loss = self.model.recon_loss(z, batch.edge_index)
        self.log('train_loss',loss, on_step=True,on_epoch=True,batch_size=z.shape[0])
        return loss
    def validation_step(self, batch, batch_idx):
        z = self(batch)
        loss = self.model.recon_loss(z, batch.edge_index)
        self.log('val_loss',loss,on_step=True,on_epoch=True,batch_size=z.shape[0])
    def test_step(self, batch, batch_idx):
        z = self(batch)
        loss = self.model.recon_loss(z, batch.edge_index)
        self.log('test_loss',loss,on_step=False,on_epoch=True,batch_size=z.shape[0])
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)

from automate import LinearBlock, BipartiteResMRConv
from torch.nn import ModuleList
from torchmetrics import Accuracy
from torch.nn.functional import cross_entropy
class CodePredictor(pl.LightningModule):
    def __init__(self, in_channels, out_channels, mlp_layers, mp_layers):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp_layers = mlp_layers
        self.mp_layers = mp_layers
        
        self.mp = ModuleList([BipartiteResMRConv(in_channels) for _ in range(mp_layers)])
        self.mlp = LinearBlock(*([in_channels]*mlp_layers), out_channels, last_linear=True)
        
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        
    def forward(self, data):
        x = data.x#torch.cat([data.x,data.z],dim=1)
        for mp in self.mp:
            x = mp(x,x,data.edge_index)
        x = self.mlp(x)
        return x
    
    def training_step(self, batch, batch_idx):
        scores = self(batch)
        preds = scores.argmax(dim=1)
        target = batch.y.reshape(-1)
        loss = cross_entropy(scores, target)
        self.train_acc(preds, target)
        batch_size = len(target)
        self.log('train_loss',loss,on_epoch=True,on_step=True,batch_size=batch_size)
        self.log('train_acc',self.train_acc,on_epoch=True,on_step=True,batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        scores = self(batch)
        preds = scores.argmax(dim=1)
        target = batch.y.reshape(-1)
        loss = cross_entropy(scores, target)
        self.val_acc(preds, target)
        batch_size = len(target)
        self.log('val_loss',loss,on_epoch=True,on_step=True,batch_size=batch_size)
        self.log('val_acc',self.val_acc,on_epoch=True,on_step=True,batch_size=batch_size)
    
    def test_step(self, batch, batch_idx):
        scores = self(batch)
        preds = scores.argmax(dim=1)
        target = batch.y.reshape(-1)
        loss = cross_entropy(scores, target)
        self.test_acc(preds, target)
        batch_size = len(target)
        self.log('test_loss',loss,on_epoch=True,on_step=False,batch_size=batch_size)
        self.log('test_acc',self.test_acc,on_epoch=True,on_step=False,batch_size=batch_size)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    @classmethod
    def get_params(cls, exp_name):
        return [64,16,2,int(exp_name[2])]


class CodedFabwave(torch.utils.data.Dataset):
    def __init__(self, index, coded_data, seed, fraction, mode = 'train'):
        super().__init__()
        
        def create_subset(data, seed, size):
            random.seed(seed)
            return random.sample(data, size)

        category_counts = Counter([key[0] for key in index['train']])
        category_encodings = {cat:i for i,cat in enumerate(sorted(category_counts.keys()))}
        category_seeds = lambda seed: {k:(v + seed*len(category_encodings)) for k,v in category_encodings.items()}
        
        def subset_category(cat, seed, fraction):
            return create_subset(
                [x for x in index['train'] if x[0] == cat], 
                category_seeds(seed)[cat], 
                max(2, ceil(fraction*category_counts[cat])))
        
        def subset_train(seed, fraction):
            return [key for cat in category_counts for key in subset_category(cat, seed, fraction)]
        
        keyset = index['test']
        if mode in ['train','validate']:
            keyset = subset_train(seed, fraction)
            
            val_size = 0.2 if ceil(0.2*len(keyset) >= 26) else 26
     
            train_keys, val_keys = train_test_split(
                keyset, test_size=val_size, random_state=42, stratify=[k[0] for k in keyset],
            )
            
            keyset = train_keys if mode == 'train' else val_keys
        
        self.all_data = [tg.data.Data(**coded_data[index['template'].format(*k)]) for k in keyset]
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        return self.all_data[idx]

class PoolingPredictor(pl.LightningModule):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.lin1 = Linear(in_size, hidden_size)
        self.lin2 = Linear(hidden_size, out_size)
        self.test_accuracy = Accuracy()
    
    def forward(self, data):
        x = self.lin1(data.x)
        x_batch = data.batch if 'batch' in data else torch.zeros_like(data.x[:,0]).long()
        x = global_max_pool(x, x_batch)
        x = leaky_relu(x)
        x = self.lin2(x)
        return x
    
    def training_step(self, batch, batch_idx):
        scores = self(batch)
        loss = cross_entropy(scores, batch.y.reshape(-1))
        bs = len(batch.y.reshape(-1))
        acc = (scores.argmax(dim=1) == batch.y).sum() / len(scores)
        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=bs)
        self.log('train_acc', acc, on_step=True, on_epoch=True, batch_size=bs)
        return loss
    
    def validation_step(self, batch, batch_idx):
        scores = self(batch)
        loss = cross_entropy(scores, batch.y.reshape(-1))
        bs = len(batch.y.reshape(-1))
        acc = (scores.argmax(dim=1) == batch.y).sum() / len(scores)
        self.log('val_loss', loss, on_step=True, on_epoch=True, batch_size=bs)
        self.log('val_acc', acc, on_step=True, on_epoch=True, batch_size=bs)
        
    def test_step(self, batch, batch_idx):
        scores = self(batch)
        preds = scores.argmax(dim=1)
        targets = batch.y.reshape(-1)
        self.test_accuracy(preds, targets)
        self.log('test_acc', self.test_accuracy, on_step=False, on_epoch=True, batch_size=len(targets))
        
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    @classmethod
    def get_params(cls, exp_name):
        return [64,64,26]


class DictDataset(torch.utils.data.Dataset):
    def __init__(self, index, data, mode='train', val_frac=.2,seed=42,undirected=True,train_size=None):
        index
        keyset = index['test']
        if mode in ['train', 'validate']:
            keyset = index['train']
            if train_size:
                keyset = create_subset(keyset, seed, train_size)
            train_keys, val_keys = train_test_split(keyset, test_size=val_frac, random_state=seed)
            keyset = train_keys if mode == 'train' else val_keys
        self.data = {i:tg.data.Data(**data[index['template'].format(*key)]) for i,key in enumerate(keyset)}
        if undirected:
            for k,v in self.data.items():
                v.edge_index = torch.cat([v.edge_index, v.edge_index[[1,0]]],dim=1)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class DictDatamodule(pl.LightningDataModule):
    def __init__(self, index, data, val_frac=0.2, seed=42, batch_size=32,train_size=None):
        super().__init__()
        self.val_frac = val_frac
        self.seed = seed
        self.ds_train = DictDataset(index, data, 'train', val_frac, seed, True, train_size)
        self.ds_val = DictDataset(index, data, 'validate', val_frac, seed, True, train_size)
        self.ds_test = DictDataset(index, data, 'test')
        self.batch_size = batch_size
    
    def train_dataloader(self):
        return DataLoader(
            self.ds_train, 
            batch_size=min(len(self.ds_train), self.batch_size), 
            shuffle=True, 
            num_workers=8, 
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val, 
            batch_size=min(len(self.ds_val),self.batch_size), 
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=1, shuffle=False)


# Functions for navigating the output

import os

# By default, assume that we want the latest version and the lowest val_loss
def load_model(
    tb_dir, 
    model_class=CodePredictor,
    v_num='last', 
    metric='val_loss', 
    descending=False, 
    device='cpu'
):  
    exp_name = os.path.split(tb_dir)[-1].split('_')[0]
    versions = {int(d.split('_')[-1]):d for d in os.listdir(tb_dir) if 'version_' in d} 
    if v_num not in versions: # Take the latest if specified version isn't found
        v_num = sorted(versions.keys(), reverse=True)[0]
    version = versions[v_num]
    checkpoint_dir = os.path.join(tb_dir, version, 'checkpoints')
    checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt') and metric in f]
    cp_by_loss = [(float(cp[cp.index(f'{metric}=') + len(f'{metric}='):-5]),cp) for cp in checkpoints]
    cp_by_loss = sorted(cp_by_loss, key=lambda x: x[0], reverse=descending)
    checkpoint = cp_by_loss[0][1]
    model = model_class(*model_class.get_params(exp_name))
    sd = torch.load(checkpoint,map_location=device)['state_dict']
    model.load_state_dict(sd)
    return model

def load_models(
    root, 
    seeds=list(range(10)),
    model_class=CodePredictor,
    size_type=int,
    v_num='last',
    metric='val_loss',
    descending=False,
    device='cpu'
):
    output = {}
    for seed in seeds:
        # exp_name : [(size, tb_path),...]
        model_dirs = find_models(root, seed, size_type)
        for exp_name, tb_paths in model_dirs.items():
            output[exp_name] = output.get(exp_name,{})
            for size, tb_path in tb_paths:
                output[exp_name][size] = output[exp_name].get(size, [])
                model = load_model(tb_path, model_class, v_num, metric, descending, device)
                output[exp_name][size].append(model)
    # output_format: exp_name: size :  [model,...] (seed order)
    return output

def find_models(root, seed = 0, size_type = int):
    model_dirs = [
        (d.split('_')[0], size_type(d.split('_')[1]), os.path.join(root,d)) 
        for d in os.listdir(root) 
        if '_' in d and os.path.isdir(os.path.join(root,d)) and int(d.split('_')[-1]) == seed
    ]
    m_dir_dict = dict()
    for model, train_size, path in model_dirs:
        ms = m_dir_dict.get(model, [])
        ms.append((train_size, path))
        m_dir_dict[model] = ms
    # Sort by train_size
    for model,paths in m_dir_dict.items():
        m_dir_dict[model] = sorted(paths, key=lambda x: x[0])
    
    return m_dir_dict

def our_fabwave_models(root):
    return load_models(
    os.path.join(root, 'fabwave_nn_pool'),
    model_class=PoolingPredictor,
    metric='step',
    descending=True,
    size_type = float
    )['fwp']

def our_f360seg_models(root):
    return load_models(os.path.join(root,'f360seg'))['mp2']

def our_mfcad_models(root):
    return load_models(os.path.join(root,'mfcad'))['mp2']

## DEPRECATED ##

def load_segmentation_models(root, seed = 0, size_type=int):
    m_dir_dict = find_models(root, seed, size_type)
    for k,v in m_dir_dict.items():
        for i,(s, p) in enumerate(v):
            v[i] = (s, load_segmentation_model(int(k[-1]), p))
    return m_dir_dict

def load_segmentation_model(mp_layers, tb_dir):
    checkpoint_dir = os.path.join(tb_dir, 'version_0', 'checkpoints')
    checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt') and 'val_loss' in f]
    model = CodePredictor(64, 16, 2, mp_layers)
    sd = torch.load(checkpoints[0],map_location='cpu')['state_dict']
    model.load_state_dict(sd)
    return model