import torch
from abc import ABC, abstractmethod
import numpy as np
import math


class Matrix(ABC):
    def __init__(
        self,
        min_val,
        max_val,
    ):
        self.min_val = min_val
        self.max_val = max_val

    @abstractmethod
    def tokenize(self, X, mask):
        pass

    @abstractmethod
    def sample(self, n_samples, m, n, r):
        pass


class RealMatrix(Matrix):
    def __init__(self, args, device):
        super(RealMatrix, self).__init__(
            min_val=args.min_val,
            max_val=args.max_val,
        )

        self.prec = args.prec
        self.device = device

    def tokenize(self, X, mask):
        X_rounded = torch.round(X, decimals=self.prec)
        X_token = (1 + (X_rounded * 10**self.prec) + (2.0 * 10**self.prec)).to(torch.int)

        X_mask_token = (X_token * mask).view(X.shape[0], -1)
        return X_token.view(X.shape[0], -1), X_mask_token

    def sample(self, n_samples, m, n, r, p_mask):
        U = -1 + 2 * torch.rand(
            size=(n_samples, m, r)
        ).to(torch.float).to(self.device)

        V = -1 + 2 * torch.rand(
            size=(n_samples, n, r)
        ).to(torch.float).to(self.device)

        matrix = U @ V.permute(0, 2, 1)

        uniform_matrix = torch.rand(matrix.shape).to(self.device)
        mask = uniform_matrix > p_mask

        matrix_tok, matrix_mask_tok = self.tokenize(matrix, mask)

        samples = (
            torch.cat(
                [
                    matrix_mask_tok,
                    404 * torch.ones(size=(n_samples, 1)).to(self.device),
                    matrix_tok,
                ],
                axis=-1,
            )
            .to(int)
            .detach()
        )
        return samples  # , torch.round(matrix, decimals=self.prec).view(matrix.shape[0], -1)
