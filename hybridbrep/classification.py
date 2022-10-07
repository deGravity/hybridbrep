import json
import random
from collections import Counter

import pandas as pd
import pytorch_lightning as pl
import torch
import torch_geometric as tg
from automate import BipartiteResMRConv, LinearBlock
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.nn import Linear, ModuleList
from torch.nn.functional import cross_entropy, leaky_relu
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_max_pool
from torchmetrics import Accuracy
from tqdm import tqdm
from math import ceil


class CodePoolingPredictor(pl.LightningModule):
    def __init__(self, in_channels, hidden_size, out_channels, mp_layers):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.mp_layers = mp_layers

        self.lin1 = Linear(in_channels, hidden_size)
        self.lin2 = Linear(hidden_size, out_channels)
        self.mp = ModuleList([BipartiteResMRConv(in_channels) for _ in range(mp_layers)])

        
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        
    def forward(self, data):
        x = data.x
        for mp in self.mp:
            x = mp(x,x,data.edge_index)
        x = self.lin1(data.x)
        x_batch = data.batch if 'batch' in data else torch.zeros_like(data.x[:,0]).long()
        x = global_max_pool(x, x_batch)
        x= leaky_relu(x)
        x = self.lin2(x)
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



def create_subset(data, seed, size):
    random.seed(seed)
    return random.sample(data, size)


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

class CodedFabwaveDatamodule(pl.LightningDataModule):
    def __init__(self, index, coded_set, seed, experiment_size):
        super().__init__()
        self.seed = seed
        self.ds_train = CodedFabwave(index, coded_set, seed, experiment_size, 'train')
        self.ds_val = CodedFabwave(index, coded_set, seed, experiment_size, 'validate')
        self.ds_test = CodedFabwave(index, coded_set, seed, experiment_size, 'test')
    
    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=1024)
    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=1024)
    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=387, shuffle=False)

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


def train_precoded_classification(
    index_path,
    precoded_path,
    logdir,
    logname,
    output_path,
    experiment_sizes = [.01, .05, .1, .2, .5, .75, 1.0], 
    seeds = list(range(10))
):
    with open(index_path, 'r') as f:
        index = json.load(f)
    coded_set = torch.load(precoded_path)

    max_epochs = 2000
    test_accuracies = []

    for experiment_size in tqdm(experiment_sizes):
        for seed in tqdm(seeds):
            exp_name = f'{logname}_{experiment_size}_{seed}'

            model = CodePoolingPredictor(64, 64, 26, 0)#(64, 16, 2, 2)

            datamodule = CodedFabwaveDatamodule(index, coded_set, seed, experiment_size)
            callbacks = [
                EarlyStopping(monitor='val_loss', mode='min', patience=100),
                ModelCheckpoint(monitor='val_loss', save_top_k=1, filename="{epoch}-{val_loss:.6f}",mode="min"),
            ]
            logger = TensorBoardLogger(logdir, exp_name)
            trainer = pl.Trainer(max_epochs=max_epochs, logger=logger, callbacks=callbacks, gpus=1)
            trainer.fit(model, datamodule)
            results = trainer.test(datamodule=datamodule)
            acc = results[0]['test_acc']
            test_accuracies.append({
                'experiment': logname,
                'train_size': experiment_size,
                'seed': seed,
                'accuracy': acc
            })
    
    accs_df = pd.DataFrame.from_records(test_accuracies)
    accs_df.to_csv(output_path)
            

