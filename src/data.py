import torch
from abc import ABC, abstractmethod
import numpy as np
import math


class Matrix(ABC):
    def __init__(self, min_val, max_val, min_vocab, max_vocab):
        # self.p_mask = p_mask
        self.min_val = min_val
        self.max_val = max_val
        self.min_vocab = min_vocab
        self.max_vocab = max_vocab
        self.vocab = {"MASK": 0}

    @abstractmethod
    def construct_vocab(self):
        pass

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
            min_vocab=args.min_vocab,
            max_vocab=args.max_vocab,
        )

        self.prec = args.prec
        self.construct_vocab()
        self.device = device

    def construct_vocab(self):
        for val in range(
            int(self.min_vocab * (10**self.prec)),
            int((self.max_vocab * 10**self.prec) + 1),
        ):
            self.vocab[str(val / (10**self.prec))] = len(self.vocab.keys())

    def tokenize(self, X, mask):
        X_rounded = torch.round(X, decimals=self.prec)
        X_mask_token = (
            1 + ((X_rounded * 10**self.prec) - (self.min_vocab * 10**self.prec))
        ).to(torch.int) * mask

        X_mask_token = X_mask_token.view(X.shape[0], -1)
        return X_mask_token, X_rounded.to(torch.float)

    def sample(self, n_samples, m, n, r, p_mask, return_uv=False):
        U = self.min_val + (self.max_val - self.min_val) * torch.rand(
            size=(n_samples, m, r)
        ).to(torch.float).to(self.device)

        V = self.min_val + (self.max_val - self.min_val) * torch.rand(
            size=(n_samples, n, r)
        ).to(torch.float).to(self.device)

        matrix = U @ V.permute(0, 2, 1)

        uniform_matrix = torch.rand(matrix.shape).to(self.device)
        mask = uniform_matrix > p_mask
        
        matrix_mask_tok, matrix_rounded = self.tokenize(matrix, mask)

        matrix_mask_tok, matrix_rounded, mask = (
            matrix_mask_tok.to(self.device),
            matrix_rounded.to(self.device),
            mask.to(self.device),
        )

        if return_uv:
            return matrix_mask_tok, matrix_rounded, mask, U, V

        return matrix_mask_tok, matrix_rounded, mask

    def sample_patch(self, n_samples, m, n, r, p_mask, return_uv=False):
        U = self.min_val + (self.max_val - self.min_val) * torch.rand(
            size=(n_samples, m, r)
        ).to(torch.float).to(self.device)

        V = self.min_val + (self.max_val - self.min_val) * torch.rand(
            size=(n_samples, n, r)
        ).to(torch.float).to(self.device)

        matrix = U @ V.permute(0, 2, 1)

        uniform_matrix = torch.rand(matrix.shape).to(self.device)
        mask = uniform_matrix > p_mask

        matrix_mask_tok, matrix_rounded = self.tokenize(matrix, mask)

        corr_matrix = -1 * matrix

        corr_matrix_mask, corr_matrix_rounded = self.tokenize(corr_matrix, mask)
        matrix_mask_tok, matrix_rounded, corr_matrix_mask, corr_matrix_rounded, mask = (
            matrix_mask_tok.to(self.device),
            matrix_rounded.to(self.device),
            corr_matrix_mask.to(self.device),
            corr_matrix_rounded.to(self.device),
            mask.to(self.device),
        )

        return (
            matrix_mask_tok,
            matrix_rounded,
            corr_matrix_mask,
            corr_matrix_rounded,
            mask,
        )


class GaussianMatrix(RealMatrix):
    def __init__(self, args, device):
        super(GaussianMatrix, self).__init__(args=args, device=device)
        self.scale = args.gaussian_scale

    def sample(self, n_samples, m, n, r, p_mask, return_uv=False):
        print("Gaussian")
        U = self.scale * torch.randn(size=(n_samples, m, r)).to(torch.float).to(
            self.device
        )

        V = self.scale * torch.randn(size=(n_samples, n, r)).to(torch.float).to(
            self.device
        )

        matrix = U @ V.permute(0, 2, 1)

        uniform_matrix = torch.rand(matrix.shape).to(self.device)
        mask = uniform_matrix > p_mask

        matrix_mask_tok, matrix_rounded = self.tokenize(matrix, mask)
        matrix_mask_tok, matrix_rounded, mask = (
            matrix_mask_tok.to(self.device),
            matrix_rounded.to(self.device),
            mask.to(self.device),
        )

        if return_uv:
            return matrix_mask_tok, matrix_rounded, mask, U, V

        return matrix_mask_tok, matrix_rounded, mask


class LaplaceMatrix(RealMatrix):
    def __init__(self, args, device):
        super(LaplaceMatrix, self).__init__(args=args, device=device)
        self.scale = args.laplace_scale

    def sample(self, n_samples, m, n, r, p_mask, return_uv=False):
        print("Laplacian")
        laplace = torch.distributions.laplace.Laplace(
            loc=0, scale=self.scale, validate_args=None
        )

        U = (
            laplace.sample(sample_shape=(n_samples, m, r))
            .to(torch.float)
            .to(self.device)
        )

        V = (
            laplace.sample(sample_shape=(n_samples, n, r))
            .to(torch.float)
            .to(self.device)
        )

        matrix = U @ V.permute(0, 2, 1)

        uniform_matrix = torch.rand(matrix.shape).to(self.device)
        mask = uniform_matrix > p_mask

        matrix_mask_tok, matrix_rounded = self.tokenize(matrix, mask)
        matrix_mask_tok, matrix_rounded, mask = (
            matrix_mask_tok.to(self.device),
            matrix_rounded.to(self.device),
            mask.to(self.device),
        )

        if return_uv:
            return matrix_mask_tok, matrix_rounded, mask, U, V

        return matrix_mask_tok, matrix_rounded, mask


