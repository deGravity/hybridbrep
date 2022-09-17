import torch
import automate
import torch_geometric as tg
import pytorch_lightning as pl
from automate import LinearBlock, SBGCN

class UVPred(pl.LightningModule):
    def __init__(self):
        super().__init__()

        f_in = 17
        l_in = 10
        e_in = 15
        v_in = 3
        sbgcn_hidden_size = 64
        k = 0
        self.sbgcn = SBGCN(f_in, l_in, e_in, v_in, sbgcn_hidden_size, 0)
        self.edge_lin = automate.LinearBlock(sbgcn_hidden_size, 512, 512, 30)

        """
        self.vert_in = LinearBlock(3,128,128)
        self.edge_in = LinearBlock(15,128,128)

        self.vert_to_edge = tg.nn.GATv2Conv(
            in_channels=(128,128),
            out_channels=128
        )
        self.edge_conv = LinearBlock(128, 512, 512, 512, 30, last_linear=True)
        """
    
    def forward(self, x):
        #e = self.vert_to_edge((self.vert_in(x.vertices), self.edge_in(x.edges)), x.edge_to_vertex[[1,0],:])
        #return torch.tanh(self.edge_conv(e))
        x_t, x_p, x_f, x_l, x_e, x_v = self.sbgcn(x)
        return torch.tanh(self.edge_lin(x_e))
    
    def training_step(self, batch, batch_idx):
        target = batch.edge_samples[:,::11,:].reshape(-1,30)
        pred = self(batch)
        loss = torch.nn.functional.mse_loss(pred, target)
        self.log('train_loss',loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        target = batch.edge_samples[:,::11,:].reshape(-1,30)
        pred = self(batch)
        loss = torch.nn.functional.mse_loss(pred, target)
        self.log('val_loss',loss)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    ds_train = automate.SimplePartDataset('E:/research/cad/data/cache/uv100/validate',mode='train')
    ds_val = automate.SimplePartDataset('E:/research/cad/data/cache/uv100/validate',mode='validate')
    dl_train = tg.loader.DataLoader(ds_train, batch_size=16, num_workers=8,shuffle=True)#,shuffle=True,num_workers=4)
    dl_val = tg.loader.DataLoader(ds_val, batch_size=16,shuffle=False,num_workers=8)

    model = UVPred()
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor='train_loss', save_top_k=1, filename="{epoch}-{train_loss:.6f}",mode="min",
        ),
        pl.callbacks.ModelCheckpoint(
            monitor='val_loss', save_top_k=1, filename="{epoch}-{val_loss:.6f}",mode="min",
        )
    ]
    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=callbacks,
        #limit_val_batches=1,
        gpus=[1],
        log_every_n_steps=1
        #gradient_clip_val=0.5
        )
    trainer.fit(model, train_dataloader=dl_train, val_dataloaders=dl_val)

if __name__ == '__main__':
    main()