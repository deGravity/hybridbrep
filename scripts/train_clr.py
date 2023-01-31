from hybridbrep.part_embedding import CLRDatamodule, CLREmbedder
import pytorch_lightning as pl
import torch
import json

def main():

    index_path = '/projects/grail/benjonesnb/cadlab/datasets/fusion360seg.json'
    data_path = '/projects/grail/benjonesnb/cadlab/datasets/fusion360seg_precoded_new_model_new_data.pt'
    
    with open(index_path, 'r') as f:
        index = json.load(f)
    data = torch.load(data_path)

    model = CLREmbedder()
    datamodule = CLRDatamodule(index, data)

    trainer = pl.Trainer()

    trainer.fit(model, datamodule)


if __name__ == '__main__':
    main()
