import torch
import torch.sparse
from einops import repeat, rearrange


class SparseOps:
    def __init__(self, dense, local_mask):
        """
        Args:
            dense: (B, N, T, T, D)
            local_mask: (T, T)
        """
        self.B, self.N, self.T, _, self.D = dense.shape
        self.local_mask = local_mask
        self.M = torch.sum(local_mask)
        
        dense = dense * local_mask[..., None]

        local_mask = repeat(local_mask, 't1 t2 -> b n t1 t2', b = self.B, n = self.N)
        
        indices = torch.nonzero(local_mask)
        values = dense[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]]

        del dense
        
        self.sparse = torch.sparse_coo_tensor(indices.transpose(0, 1), values, size=(self.B, self.N, self.T, self.T, self.D)).coalesce()
        self.indices = self.sparse.indices()

    def reshape_to_dense(self):
        """ transform B x N x T x T x D sparse B x N x M x D dense, where M is the number of 1 in local mask. """
        flatten_matrix = self.sparse.values()

        dense = rearrange(flatten_matrix, '(b n m) d -> b n m d', b=self.B, n=self.N)

        return dense

    def update_sparse(self, values):
        """ update the sparse matrix using new values
        
        Args:
            values (torch.Tensor): (B, N, M, D)
        
        Returns:
            torch.sparse.Tensor: 形状为 B x N x T x T x D 的稀疏张量
        """
        B, N, M = values.shape[:3]
        assert self.B == B and self.N == N and self.M == M
        new_values = values.view(-1, self.D)
        self.sparse = torch.sparse_coo_tensor(self.indices, new_values, size=(self.B, self.N, self.T, self.T, self.D)).coalesce()      
        
    def to(self, device):
        self.sparse = self.sparse.to(device)
        self.indices = self.sparse.indices()


def create_local_mask(T, lwidth):
    """ create local mask as adjacency matrix """
    local_mask = torch.zeros(T, T)

    k = (lwidth - 1) // 2 

    indices = torch.arange(T).unsqueeze(0)
    range_mask = (indices - indices.T).abs() <= k

    local_mask[range_mask] = 1

    return local_mask
