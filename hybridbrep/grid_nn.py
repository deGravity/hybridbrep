import os
import platform
if platform.system() != 'Windows':
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

from automate import SBGCN, LinearBlock, ArgparseInitialized
from .uvgrid import surface_metric, surface_normals, arc_lengths, cos_corner_angles, render_batch


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pyrender

import torch
from typing import Tuple



class FixedGridPredictor(pl.LightningModule, ArgparseInitialized):

    def get_callbacks(self):
        return [
        ModelCheckpoint(
            monitor='train/loss', save_top_k=1, filename="{epoch}-{train_loss:.6f}",mode="min",
        ),
        ModelCheckpoint(
            monitor='val/loss', save_top_k=1, filename="{epoch}-{val_loss:.6f}",mode="min",
        ),
        ModelCheckpoint(save_last=True),
        EarlyStopping(
            monitor=f'val/loss', patience=self.patience ,mode='min', check_finite=True
        )
    ]

    def __init__(self,
        sbgcn_hidden_size: int = 128,
        surf_hidden_sizes: Tuple[int, ...] = (1024, 1024),
        curve_hidden_sizes: Tuple[int, ...] = (1024, 1024),
        w_curve_loss: float = 1.0,
        w_surf_loss: float = 1.0,
        w_norm_loss: float = 1.0,
        w_metric_loss: float = 1.0,
        w_arc_loss: float = 1.0,
        w_corner_loss: float = 1.0,
        sbgcn_k: int = 0,
        render_size: Tuple[int, int] = (800,800),
        render_count: int = 5,
        num_samples: int = 10,
        learning_rate: float = 1e-3,
        patience: int = 3
    ):
        super().__init__()

        f_in = 17
        l_in = 10
        e_in = 15
        v_in = 3

        self.N = num_samples

        self.patience = patience


        self.w_curve_loss = w_curve_loss
        self.w_surf_loss = w_surf_loss
        self.w_norm_loss = w_norm_loss
        self.w_metric_loss = w_metric_loss
        self.w_arc_loss = w_arc_loss
        self.w_corner_loss = w_corner_loss

        self.learning_rate = learning_rate

        self.sbgcn_hidden_size = sbgcn_hidden_size
        self.sbgcn_k = sbgcn_k
        self.sbgcn = SBGCN(f_in, l_in, e_in, v_in, self.sbgcn_hidden_size, self.sbgcn_k)
        self.surf_hidden_sizes = surf_hidden_sizes
        self.curve_hidden_sizes = curve_hidden_sizes

        self.render_size = render_size
        #self.renderer = pyrender.OffscreenRenderer(*self.render_size)
        self.render_count = render_count

        self.curve_lin = LinearBlock(self.sbgcn_hidden_size, *self.curve_hidden_sizes, 3*self.N)
        self.surf_lin = LinearBlock(self.sbgcn_hidden_size, *self.surf_hidden_sizes, 4*self.N*self.N)
        
        self.save_hyperparameters(ignore=['render_size', 'render_count'])

    #def __del__(self):
        #self.renderer.delete()
    
    def forward(self, x):
        _, _, x_f, _, x_e, _ = self.sbgcn(x)
        surf_pred = torch.tanh(self.surf_lin(x_f))
        curve_pred = torch.tanh(self.curve_lin(x_e))
        return surf_pred.reshape(-1,self.N,self.N,4), curve_pred.reshape(-1,self.N,3)
    
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, 
            {
            'val/curve_loss': 1.0,
            'val/surf_loss': 1.0,
            'val/norm_loss': 1.0,
            'val/metric_loss': 1.0,
            'val/arc_loss': 1.0,
            'val/loss': 1.0
            })

    def common_step(self, batch, batch_idx, step='train'):
        curve_target = batch.edge_samples
        surf_target = batch.face_samples
        surf_pred, curve_pred = self(batch)

        if self.w_curve_loss > 0:
            curve_loss = torch.nn.functional.mse_loss(curve_pred, curve_target)
        else:
            curve_loss = torch.tensor(1.0)

        if self.w_surf_loss > 0:
            surf_loss = torch.nn.functional.mse_loss(surf_pred, surf_target)
        else:
            surf_loss = torch.tensor(1.0)

        if self.w_norm_loss > 0:
            norm_target = surface_normals(surf_target)
            norm_pred = surface_normals(surf_pred)
            norm_loss = torch.nn.functional.mse_loss(norm_target, norm_pred)
        else:
            norm_loss = torch.tensor(1.0)

        if self.w_metric_loss > 0:
            target_du, target_dv = surface_metric(surf_target)
            pred_du, pred_dv = surface_metric(surf_pred)
            metric_loss_u = torch.nn.functional.mse_loss(pred_du, target_du)
            metric_loss_v = torch.nn.functional.mse_loss(pred_du, target_du)
            metric_loss = metric_loss_u / 2 + metric_loss_v / 2
        else:
            metric_loss = torch.tensor(1.0)

        if self.w_arc_loss > 0:
            arc_target = arc_lengths(curve_target)
            arc_pred = arc_lengths(curve_pred)
            arc_loss = torch.nn.functional.mse_loss(arc_pred, arc_target)
        else:
            arc_loss = torch.tensor(1.0)

        if self.w_corner_loss > 0:
            corner_target = cos_corner_angles(curve_target)
            corner_pred = cos_corner_angles(curve_pred)
            corner_loss = torch.nn.functional.mse_loss(corner_pred, corner_target)
        else:
            corner_loss = torch.tensor(1.0)

        loss = (
            self.w_surf_loss * surf_loss +
            self.w_curve_loss * curve_loss +
            self.w_metric_loss * metric_loss + 
            self.w_norm_loss * norm_loss + 
            self.w_arc_loss * arc_loss + 
            self.w_corner_loss * corner_loss
        ) / (
            self.w_surf_loss +
            self.w_curve_loss +
            self.w_metric_loss + 
            self.w_norm_loss + 
            self.w_arc_loss + 
            self.w_corner_loss
        )

        output_dict = {
            f'{step}/curve_loss': curve_loss,
            f'{step}/surf_loss': surf_loss,
            f'{step}/norm_loss': norm_loss,
            f'{step}/metric_loss': metric_loss,
            f'{step}/arc_loss': arc_loss,
            f'{step}/loss': loss
        }

        return output_dict, surf_pred, curve_pred
    
    def training_step(self, batch, batch_idx):
        losses, _, _ = self.common_step(batch, batch_idx, 'train')
        batch_size = len(batch.face_to_flat_topos[1].unique())
        self.log_dict(losses, batch_size=batch_size, on_step=True, on_epoch=True)
        return losses['train/loss']
    
    def on_validation_start(self):
        self.renderer = pyrender.OffscreenRenderer(*self.render_size)

    def on_validation_end(self):
        self.renderer.delete()

    def validation_step(self, batch, batch_idx):
        losses, surf_pred, curve_pred = self.common_step(batch, batch_idx, 'val')
        batch_size = len(batch.face_to_flat_topos[1].unique())
        self.log_dict(losses, batch_size=batch_size)
        if batch_idx == 0 :
            tensorboard = self.logger.experiment
            comparison = render_batch(
                batch, 
                (surf_pred, curve_pred), 
                self.render_count, 
                self.renderer
            )
            tensorboard.add_image('val/comparison', comparison, self.current_epoch)
    
    def test_step(self, batch, batch_idx):
        losses, surf_pred, curve_pred = self.common_step(batch, batch_idx, 'test')
        return losses

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
