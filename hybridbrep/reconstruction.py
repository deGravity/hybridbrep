import pytorch_lightning as pl
import torch
import torch_geometric as tg
from automate import BipartiteResMRConv, LinearBlock
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from .implicit import ImplicitDecoder
from .hybridpart import HybridPartDataset


class BRepFaceEncoder(torch.nn.Module):
    def __init__(self, hidden_size, use_loops=True):
        super().__init__()
        self.use_loops = use_loops
        self.v_in = LinearBlock(3, hidden_size)
        self.e_in = LinearBlock(18, hidden_size)
        if self.use_loops:
            self.l_in = LinearBlock(10, hidden_size)
        self.f_in = LinearBlock(21, hidden_size)
        self.v_to_e = BipartiteResMRConv(hidden_size)
        if self.use_loops:
            self.e_to_l = BipartiteResMRConv(hidden_size)
            self.l_to_f = BipartiteResMRConv(hidden_size)
        else:
            self.e_to_f = BipartiteResMRConv(hidden_size)
    def forward(self, data):
        v = self.v_in(data.vertices)
        e = data.edges#torch.cat([data.edge_curves, data.edge_curve_parameters, data.edge_curve_flipped.reshape((-1,1))], dim=1)
        e = self.e_in(e)
        if self.use_loops:
            l = self.l_in(data.loops)
        f = data.faces#torch.cat([data.face_surfaces, data.face_surface_parameters, data.face_surface_flipped.reshape((-1,1))], dim=1)
        f = self.f_in(f)
        # TODO - incorporate edge-loop data and vert-edge data
        # Potential TODO - heterogenous input of edges and curves based on function type
        e = self.v_to_e(v, e, data.vertex_to_edge)
        if self.use_loops:
            l = self.e_to_l(e, l, data.edge_to_loop)
            f = self.l_to_f(l, f, data.loop_to_face)
        else:
            f = self.e_to_f(e, f, data.edge_to_face)
        return f

class BRepFaceAutoencoder(pl.LightningModule):
    def __init__(self, code_size, hidden_size, decoder_layers, use_loops=True):
        super().__init__()
        self.encoder = BRepFaceEncoder(code_size,use_loops)
        self.decoder = ImplicitDecoder(code_size+2, 4, hidden_size, decoder_layers, use_tanh=False)
    
    def forward(self, data, uv, uv_idx):
        codes = self.encoder(data)
        indexed_codes = codes[uv_idx]
        uv_codes = torch.cat([uv, indexed_codes], dim=1)
        pred = self.decoder(uv_codes)
        return pred
    
    def encode(self, data):
        with torch.no_grad():
            return self.encoder(data)
    
    def testing_loss(self, data):
        with torch.no_grad():
            codes = self.encoder(data)
            repeated_codes = codes.repeat_interleave(data.surface_coords.shape[1],dim=0)
            uvs = data.surface_coords.reshape((-1,2))
            uv_codes = torch.cat([uvs, repeated_codes],dim=1)
            target = torch.cat([data.surface_samples[:,:,:3],data.surface_samples[:,:,-1].unsqueeze(-1)],dim=-1)
            target[torch.isinf(target)] = -0.01
            pred = self.decoder(uv_codes).reshape_as(target)
            loss = torch.nn.functional.mse_loss(pred, target)
        return loss, len(codes)

    def training_step(self, data, batch_idx):
        codes = self.encoder(data)
        repeated_codes = codes.repeat_interleave(data.surface_coords.shape[1],dim=0)
        uvs = data.surface_coords.reshape((-1,2))
        uv_codes = torch.cat([uvs, repeated_codes],dim=1)
        target = torch.cat([data.surface_samples[:,:,:3],data.surface_samples[:,:,-1].unsqueeze(-1)],dim=-1)
        target[torch.isinf(target)] = -0.01
        pred = self.decoder(uv_codes).reshape_as(target)
        loss = torch.nn.functional.mse_loss(pred, target)
        batch_size = data.surface_coords.shape[0]
        self.log('train_loss', loss, batch_size=batch_size, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, data, batch_idx):
        codes = self.encoder(data)
        repeated_codes = codes.repeat_interleave(data.surface_coords.shape[1],dim=0)
        uvs = data.surface_coords.reshape((-1,2))
        uv_codes = torch.cat([uvs, repeated_codes],dim=1)
        target = torch.cat([data.surface_samples[:,:,:3],data.surface_samples[:,:,-1].unsqueeze(-1)],dim=-1)
        pred = self.decoder(uv_codes).reshape_as(target)
        loss = torch.nn.functional.mse_loss(pred, target)
        batch_size = data.surface_coords.shape[0]
        self.log('val_loss', loss, batch_size=batch_size, on_epoch=True, on_step=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def visualize_losses(self, data):
        codes = self.encoder(data)
        repeated_codes = codes.repeat_interleave(data.surface_coords.shape[1],dim=0)
        uvs = data.surface_coords.reshape((-1,2))
        uv_codes = torch.cat([uvs, repeated_codes], dim=1)
        target = data.surface_samples[:,:,-1]
        pred = self.decoder(uv_codes).reshape_as(target)
        residual = pred - target



def train_reconstruction(index_path, preprocessed_path, logdir, logname, embedding_size=64, hidden_size=1024, layers=4, patience=8, use_loops=False):
    ds = HybridPartDataset(index_path, preprocessed_path, mode='train')
    #ds = BRepDS('D:/fusion360segmentation/simple_train_test.json', 'D:/fusion360segmentation/simple_preprocessed', 'train')
    print(f'Train Set size = {len(ds)}')
    ds_val = HybridPartDataset(index_path, preprocessed_path, mode='validate')
    #ds_val = BRepDS('D:/fusion360segmentation/simple_train_test.json', 'D:/fusion360segmentation/simple_preprocessed', 'validate')
    print(f'Val Set Size = {len(ds_val)}')
    dl = tg.loader.DataLoader(ds, batch_size=8, shuffle=True, num_workers=8, persistent_workers=True)
    dl_val = tg.loader.DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=8, persistent_workers=True)
    model = BRepFaceAutoencoder(embedding_size, hidden_size,layers,use_loops)
    #sd = torch.load('best-gen-implicit-weights.ckpt')['state_dict']
    #model.load_state_dict(sd)
    callbacks = [
            pl.callbacks.ModelCheckpoint(
                monitor='train_loss', save_top_k=1, filename="{epoch}-{train_loss:.6f}",mode="min",
            ),
            pl.callbacks.ModelCheckpoint(
                monitor='val_loss', save_top_k=1, filename="{epoch}-{val_loss:.6f}",mode="min",
            ),
            pl.callbacks.early_stopping.EarlyStopping(
                    monitor='val_loss', mode='min', patience=patience
                )
        ]
    logger = TensorBoardLogger(logdir,logname)
    trainer = pl.Trainer(gpus=1, max_epochs=-1, track_grad_norm=2, callbacks=callbacks, logger=logger, gradient_clip_val=0.5)
    trainer.fit(model, dl, val_dataloaders=dl_val)
