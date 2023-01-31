import json
import random

import pandas as pd
import pytorch_lightning as pl
import torch
import torch_geometric as tg
from automate import BipartiteResMRConv, LinearBlock
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.nn import ModuleList
from torch.nn.functional import cross_entropy
from torch_geometric.loader import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm

from typing import Optional, Union, Tuple, List

from torch import Tensor
from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.utils.num_nodes import maybe_num_nodes


class PartEmbedder(torch.nn.Module):
    def __init__(self, n_layers : int = 3, in_size : int = 64, out_size : int = 128):
        super().__init__()

        self.n_layers = n_layers
        self.in_size = in_size
        self.out_size = out_size
        self.n_layers = n_layers
        
        convolutions = []
        for _ in range(n_layers):
            convolutions.append(
                tg.nn.GINConv(
                    nn = torch.nn.Sequential(
                        torch.nn.Linear(in_size, in_size),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(in_size, in_size),
                        torch.nn.LeakyReLU()
                    ),
                    train_eps=True
                )
            )
        self.graph_convs = torch.nn.ModuleList(convolutions)
        self.input_linear_proj = torch.nn.Linear(in_size, out_size)
        self.pool_projs = torch.nn.ModuleList([torch.nn.Linear(in_size, out_size) for _ in range(n_layers)])

        self.node_dropout = torch.nn.Dropout(0.3)
        self.part_dropout = torch.nn.Dropout(0.5)


    def forward(self, data):
        x = data.x
        batch = data.batch if 'batch' in data else torch.zeros_like(x[:,0])
        batch_size = data.ptr.numel() - 1 if 'ptr' in data else batch.max() + 1
        hierarchical_features = [self.input_linear_proj(tg.nn.global_max_pool(x, batch, batch_size))]
        for i in range(self.n_layers):
            graph_conv = self.graph_convs[i]
            pool_proj = self.pool_projs[i]
            x = graph_conv(x, data.edge_index)
            x_pooled = tg.nn.global_max_pool(x, batch, batch_size)
            hierarchical_features.append(pool_proj(x_pooled))
        part_feature =  self.part_dropout(torch.stack(hierarchical_features,dim=0)).sum(dim=0)
        node_features = self.node_dropout(x)
        return part_feature, node_features

class NTXentLoss(pl.LightningModule):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = torch.nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool, device=self.device)
        mask = mask.fill_diagonal_(0)
        mask[:batch_size, batch_size:] = mask[:batch_size, :batch_size]
        mask[batch_size:, :batch_size] = mask[:batch_size, :batch_size]
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples
        within a minibatch as negative examples.
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size
        mask = self.mask_correlated_samples(batch_size)

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N, device=positive_samples.device, dtype=torch.long)
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

class CLREmbedder(pl.LightningModule):
    def __init__(self, n_layers : int = 3, in_size : int = 64, out_size : int = 128, temperature : float = 0.1, **kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.n_layers = n_layers
        self.in_size = in_size
        self.out_size = out_size
        self.n_layers = n_layers
        self.temperature = temperature

        self.embedding_network = PartEmbedder(n_layers, in_size, out_size)
    
    def forward(self, data):
        return self.embedding_network(data)

    def step(self, batch, batch_idx):
        featuresA, _ = self.embedding_network(batch[0])
        featuresB, _ = self.embedding_network(batch[1])

        features = torch.cat([featuresA, featuresB], dim = 0)
        batch_size = featuresA.shape[0]
        labels = torch.cat([torch.arange(batch_size, device=features.device) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        features = torch.nn.functional.normalize(features, dim = 1)

        similarity_matrix = features @ features.T

        mask = torch.eye(labels.shape[0], dtype=torch.bool, device=features.device)

        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        loss = torch.nn.functional.cross_entropy(logits, labels)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=batch[0].x.shape[0])
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('val_loss', loss, on_step=False, on_epoch=True, batch_size=batch[0].x.shape[0])
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def create_subset(data, seed, size):
    random.seed(seed)
    return random.sample(data, size)

class CLRDataset(torch.utils.data.Dataset):
    def __init__(self, 
        index : dict, 
        data : dict, 
        mode : str = 'train', 
        val_frac : float =.2,
        seed : int = 42,
        train_size : Optional[ Union[int, float]] = None,
        id_prob : float = 0.1,
        n_hops : int = 1, 
        node_dropout : float = .4, 
        edge_dropout : float = .4
    ):
        keyset = index['test']
        if mode in ['train', 'validate']:
            keyset = index['train']
            if train_size:
                keyset = create_subset(keyset, seed, train_size)
            train_keys, val_keys = train_test_split(keyset, test_size=val_frac, random_state=seed)
            keyset = train_keys if mode == 'train' else val_keys

        self.data = {i:tg.data.Data(**data[index['template'].format(*key)]) for i,key in enumerate(keyset)}
        
        for k,v in self.data.items():
            self.data[k].edge_index = torch.cat([v.edge_index, v.edge_index[[1,0]]],dim=1)
        self.id_prob = id_prob
        self.n_hops = n_hops
        self.node_dropout = node_dropout
        self.edge_dropout = edge_dropout
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        T1 = [self.connected_patch, self.drop_nodes, self.drop_edges][random.randint(0,2)]
        T2 = [self.connected_patch, self.drop_nodes, self.drop_edges][random.randint(0,2)]
        
        if random.random() <= self.id_prob:
            T1 = self.identity
        
        d1 = T1(self.data[idx])
        
        d2 = T2(self.data[idx])

        return d1, d2

    def check_data(self, data):
        num_nodes = data.x.shape[0] if data.x.numel() > 0 else 0
        max_edge = data.edge_index.max().item() if data.edge_index.numel() > 0 else 0
        assert(max_edge < num_nodes if num_nodes > 0 else max_edge == 0)

    def identity(self, data):
        data_out = tg.data.Data(data.x, data.edge_index) # Remove y
        return data_out

    # Connected Patch - Random node + 1 or 2 hop neighborhood
    def connected_patch(self, data):
        # Randomly select nodes
        n_nodes = data.x.shape[0]
        center_node = torch.randint(low = 0, high = n_nodes, size= (1,), device=data.x.device)
        x_idx, edge_index, _, _ = k_hop_subgraph(
            center_node, 
            self.n_hops, 
            data.edge_index, 
            relabel_nodes = True,
            num_nodes = n_nodes
        )
        
        data_out = tg.data.Data(data.x[x_idx], edge_index)
        
        #self.check_data(data_out)

        return data_out
    
    def drop_nodes(self, data):
        edge_index, edge_mask, node_mask = dropout_node(data.edge_index, self.node_dropout, num_nodes = data.x.shape[0])
        
        data_out = tg.data.Data(data.x[node_mask], edge_index)

        #self.check_data(data_out)

        return data_out
    
    def drop_edges(self, data):
        edge_index, _ = dropout_edge(data.edge_index, self.edge_dropout, force_undirected=True)

        data_out = tg.data.Data(data.x, edge_index)

        #self.check_data(data_out)

        return data_out


class CLRDatamodule(pl.LightningDataModule):
    def __init__(self, index, data, val_frac=0.2, seed=42, batch_size=256,train_size=None, n_workers = 8, persistent_workers=True):
        super().__init__()
        self.val_frac = val_frac
        self.seed = seed
        self.ds_train = CLRDataset(index, data, 'train', val_frac, seed, train_size)
        self.ds_val = CLRDataset(index, data, 'validate', val_frac, seed, train_size)
        self.ds_test = CLRDataset(index, data, 'test')
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.persistent_workers = persistent_workers if n_workers > 0 else False
    
    def train_dataloader(self):
        return DataLoader(
            self.ds_train, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.n_workers, 
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val, 
            batch_size=self.batch_size, 
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False)


# Newer than our version of torch geometric
def dropout_node(edge_index: Tensor, p: float = 0.5,
                 num_nodes: Optional[int] = None,
                 training: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Randomly drops nodes from the adjacency matrix
    :obj:`edge_index` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
    indicating which edges were retained. (3) the node mask indicating
    which nodes were retained.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`BoolTensor`, :class:`BoolTensor`)

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, edge_mask, node_mask = dropout_node(edge_index)
        >>> edge_index
        tensor([[0, 1],
                [1, 0]])
        >>> edge_mask
        tensor([ True,  True, False, False, False, False])
        >>> node_mask
        tensor([ True,  True, False, False])
    """
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if not training or p == 0.0:
        node_mask = edge_index.new_ones(num_nodes, dtype=torch.bool)
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask, node_mask

    prob = torch.rand(num_nodes, device=edge_index.device)
    node_mask = prob > p
    edge_index, _, edge_mask = subgraph(node_mask, edge_index,
                                        num_nodes=num_nodes,
                                        return_edge_mask=True,
                                        relabel_nodes=True)
    return edge_index, edge_mask, node_mask

def dropout_edge(edge_index: Tensor, p: float = 0.5,
                 force_undirected: bool = False,
                 training: bool = True) -> Tuple[Tensor, Tensor]:
    r"""Randomly drops edges from the adjacency matrix
    :obj:`edge_index` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
    or index indicating which edges were retained, depending on the argument
    :obj:`force_undirected`.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        force_undirected (bool, optional): If set to :obj:`True`, will either
            drop or keep both edges of an undirected edge.
            (default: :obj:`False`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`BoolTensor` or :class:`LongTensor`)

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, edge_mask = dropout_edge(edge_index)
        >>> edge_index
        tensor([[0, 1, 2, 2],
                [1, 2, 1, 3]])
        >>> edge_mask # masks indicating which edges are retained
        tensor([ True, False,  True,  True,  True, False])

        >>> edge_index, edge_id = dropout_edge(edge_index,
        ...                                    force_undirected=True)
        >>> edge_index
        tensor([[0, 1, 2, 1, 2, 3],
                [1, 2, 3, 0, 1, 2]])
        >>> edge_id # indices indicating which edges are retained
        tensor([0, 2, 4, 0, 2, 4])
    """
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask

    row, col = edge_index

    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= p

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask

def subgraph(
    subset: Union[Tensor, List[int]],
    edge_index: Tensor,
    edge_attr: OptTensor = None,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    return_edge_mask: bool = False,
) -> Union[Tuple[Tensor, OptTensor], Tuple[Tensor, OptTensor, OptTensor]]:
    r"""Returns the induced subgraph of :obj:`(edge_index, edge_attr)`
    containing the nodes in :obj:`subset`.

    Args:
        subset (LongTensor, BoolTensor or [int]): The nodes to keep.
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        return_edge_mask (bool, optional): If set to :obj:`True`, will return
            the edge mask to filter out additional edge features.
            (default: :obj:`False`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        ...                            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5]])
        >>> edge_attr = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        >>> subset = torch.tensor([3, 4, 5])
        >>> subgraph(subset, edge_index, edge_attr)
        (tensor([[3, 4, 4, 5],
                [4, 3, 5, 4]]),
        tensor([ 7.,  8.,  9., 10.]))

        >>> subgraph(subset, edge_index, edge_attr, return_edge_mask=True)
        (tensor([[3, 4, 4, 5],
                [4, 3, 5, 4]]),
        tensor([ 7.,  8.,  9., 10.]),
        tensor([False, False, False, False, False, False,  True,
                True,  True,  True,  False, False]))
    """

    device = edge_index.device

    if isinstance(subset, (list, tuple)):
        subset = torch.tensor(subset, dtype=torch.long, device=device)

    if subset.dtype == torch.bool or subset.dtype == torch.uint8:
        num_nodes = subset.size(0)
    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        subset = index_to_mask(subset, size=num_nodes)

    node_mask = subset
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    if relabel_nodes:
        node_idx = torch.zeros(node_mask.size(0), dtype=torch.long,
                               device=device)
        node_idx[subset] = torch.arange(subset.sum().item(), device=device)
        edge_index = node_idx[edge_index]

    if return_edge_mask:
        return edge_index, edge_attr, edge_mask
    else:
        return edge_index, edge_attr

def k_hop_subgraph(
    node_idx: Union[int, List[int], Tensor],
    num_hops: int,
    edge_index: Tensor,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    flow: str = 'source_to_target',
    directed: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Computes the induced subgraph of :obj:`edge_index` around all nodes in
    :attr:`node_idx` reachable within :math:`k` hops.

    The :attr:`flow` argument denotes the direction of edges for finding
    :math:`k`-hop neighbors. If set to :obj:`"source_to_target"`, then the
    method will find all neighbors that point to the initial set of seed nodes
    in :attr:`node_idx.`
    This mimics the natural flow of message passing in Graph Neural Networks.

    The method returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.

    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central seed
            node(s).
        num_hops (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (str, optional): The flow direction of :math:`k`-hop aggregation
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            (default: :obj:`"source_to_target"`)
        directed (bool, optional): If set to :obj:`False`, will include all
            edges between all sampled nodes. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 2, 3, 4, 5],
        ...                            [2, 2, 4, 4, 6, 6]])

        >>> # Center node 6, 2-hops
        >>> subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        ...     6, 2, edge_index, relabel_nodes=True)
        >>> subset
        tensor([2, 3, 4, 5, 6])
        >>> edge_index
        tensor([[0, 1, 2, 3],
                [2, 2, 4, 4]])
        >>> mapping
        tensor([4])
        >>> edge_mask
        tensor([False, False,  True,  True,  True,  True])
        >>> subset[mapping]
        tensor([6])

        >>> edge_index = torch.tensor([[1, 2, 4, 5],
        ...                            [0, 1, 5, 6]])
        >>> (subset, edge_index,
        ...  mapping, edge_mask) = k_hop_subgraph([0, 6], 2,
        ...                                       edge_index,
        ...                                       relabel_nodes=True)
        >>> subset
        tensor([0, 1, 2, 4, 5, 6])
        >>> edge_index
        tensor([[1, 2, 3, 4],
                [0, 1, 4, 5]])
        >>> mapping
        tensor([0, 5])
        >>> edge_mask
        tensor([True, True, True, True])
        >>> subset[mapping]
        tensor([0, 6])
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True

    if not directed:
        edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask