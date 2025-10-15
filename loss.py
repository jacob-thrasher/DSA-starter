import numpy as np
import torch
from torch import nn
from torch import Tensor


def _reduction(loss: Tensor, reduction: str = 'mean') -> Tensor:
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        raise ValueError(f"`reduction` = {reduction} is not valid. Use 'none', 'mean' or 'sum'.")


class NLLLoss(nn.Module):
    """
    Adapted from: https://github.com/havakv/pycox/blob/master/pycox/models/loss.py
    Negative log-likelihood for the PMF parametrized model [1].

    Arguments:
        phi {torch.tensor} -- Estimates in (-inf, inf), where pmf = somefunc(phi).
        idx_durations {torch.tensor} -- Event times represented as indices.
        events {torch.tensor} -- Indicator of event (1.) or censoring (0.).
            Same length as 'idx_durations'.
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.

    Returns:
        torch.tensor -- The negative log-likelihood.
    """
    def __init__(self, reduction='mean', device='cuda'):
            super(NLLLoss, self).__init__()
            self.reduction = reduction
            self.device = device

    def _reduction(self, loss: Tensor, reduction: str = 'mean') -> Tensor:
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        raise ValueError(f"`reduction` = {reduction} is not valid. Use 'none', 'mean' or 'sum'.")


    def nll_pmf(self, phi: Tensor, idx_durations: Tensor, events: Tensor, reduction: str = 'mean', epsilon: float = 1e-7) -> Tensor:
        if phi.shape[1] <= idx_durations.max():
            raise ValueError(f"Network output `phi` is too small for `idx_durations`."+
                            f" Need at least `phi.shape[1] = {idx_durations.max().item()+1}`,"+
                            f" but got `phi.shape[1] = {phi.shape[1]}`")
        if events.dtype is torch.bool:
            events = events.float()
        events = events.view(-1)
        idx_durations = idx_durations.view(-1, 1)
        gamma = phi.max(1)[0]
        cumsum = phi.sub(gamma.view(-1, 1)).exp().cumsum(1)
        sum_ = cumsum[:, -1]

        part1 = phi.gather(1, idx_durations).view(-1).sub(gamma).mul(events)
        part2 = - sum_.relu().add(epsilon).log()
        part3 = sum_.sub(cumsum.gather(1, idx_durations).view(-1)).relu().add(epsilon).log().mul(1. - events)
        # need relu() in part3 (and possibly part2) because cumsum on gpu has some bugs and we risk getting negative numbers.
        loss = - part1.add(part2).add(part3)
        return self._reduction(loss, reduction)

    def forward(self, phi: Tensor, idx_durations: Tensor, events: Tensor) -> Tensor:
        '''
        Arguments:
            phi {torch.tensor} -- Predictions as float tensor with shape [batch, n_durations]
                all in (-inf, inf).
            idx_durations {torch.tensor} -- Int tensor with index of durations.
            events {torch.tensor} -- Float indicator of event or censoring (1 is event).
            rank_mat {torch.tensor} -- See pair_rank_mat function.
        '''

        return self.nll_pmf(phi, idx_durations, events, self.reduction)
    

class RankingLoss(nn.NLLLoss):
    def __init__(self, sigma=0.1, device='cuda'):
        super(RankingLoss, self).__init__()
        self.sigma = sigma
        self.device = device


    def _pair_rank_mat(self, mat, idx_durations, events, dtype='float32'):
        n = len(idx_durations)
        for i in range(n):
            dur_i = idx_durations[i]
            ev_i = events[i]
            if ev_i == 0:
                continue
            for j in range(n):
                dur_j = idx_durations[j]
                ev_j = events[j]
                if (dur_i < dur_j) or ((dur_i == dur_j) and (ev_j == 0)):
                    mat[i, j] = 1
        return mat

    def pair_rank_mat(self, idx_durations, events, dtype='float32'):
        """        
        Indicator matrix R with R_ij = 1{T_i < T_j and D_i = 1}.
        So it takes value 1 if we observe that i has an event before j and zero otherwise.
        
        Arguments:
            idx_durations {np.array} -- Array with durations.
            events {np.array} -- Array with event indicators.
        
        Keyword Arguments:
            dtype {str} -- dtype of array (default: {'float32'})
        
        Returns:
            np.array -- n x n matrix indicating if i has an observerd event before j.
        """
        idx_durations = idx_durations.reshape(-1)
        events = events.reshape(-1)
        n = len(idx_durations)
        mat = np.zeros((n, n), dtype=dtype)
        mat = self._pair_rank_mat(mat, idx_durations, events, dtype)
        return mat  

    def _diff_cdf_at_time_i(self, pmf: Tensor, y: Tensor) -> Tensor:
        """
        R is the matrix from the DeepHit code giving the difference in CDF between individual
        i and j, at the event time of j. 
        I.e: R_ij = F_i(T_i) - F_j(T_i)
        
        Arguments:
            pmf {torch.tensor} -- Matrix with probability mass function pmf_ij = f_i(t_j)
            y {torch.tensor} -- Matrix with indicator of duration/censor time.
        
        Returns:
            torch.tensor -- R_ij = F_i(T_i) - F_j(T_i)
        """
        n = pmf.shape[0]
        ones = torch.ones((n, 1), device=pmf.device)
        r = pmf.cumsum(1).matmul(y.transpose(0, 1))
        diag_r = r.diag().view(1, -1)
        r = ones.matmul(diag_r) - r
        return r.transpose(0, 1)

    def _rank_loss_deephit(self, pmf: Tensor, y: Tensor, rank_mat: Tensor, sigma: float,
                        reduction: str = 'mean') -> Tensor:
        """Ranking loss from DeepHit.
        
        Arguments:
            pmf {torch.tensor} -- Matrix with probability mass function pmf_ij = f_i(t_j)
            y {torch.tensor} -- Matrix with indicator of duration and censoring time. 
            rank_mat {torch.tensor} -- See pair_rank_mat function.
            sigma {float} -- Sigma from DeepHit paper, chosen by you.
        
        Returns:
            torch.tensor -- loss
        """
        r = self._diff_cdf_at_time_i(pmf, y)
        loss = rank_mat * torch.exp(-r/sigma)
        loss = loss.mean(1, keepdim=True)
        return _reduction(loss, reduction)

    def forward(self, phi: Tensor, idx_durations: Tensor, events: Tensor, reduction: str = 'mean') -> Tensor:
        """Rank loss proposed by DeepHit authors [1] for a single risks.

        Arguments:
            phi {torch.tensor} -- Predictions as float tensor with shape [batch, n_durations]
                all in (-inf, inf).
            idx_durations {torch.tensor} -- Int tensor with index of durations.
            events {torch.tensor} -- Float indicator of event or censoring (1 is event).
            rank_mat {torch.tensor} -- See pair_rank_mat function.
            sigma {float} -- Sigma from DeepHit paper, chosen by you.
        
        Keyword Arguments:
            reduction {string} -- How to reduce the loss.
                'none': No reduction.
                'mean': Mean of tensor.
                'sum': sum.
        
        Returns:
            torch.tensor -- Rank loss.

        References:
        [1] Changhee Lee, William R Zame, Jinsung Yoon, and Mihaela van der Schaar. Deephit: A deep learning
            approach to survival analysis with competing risks. In Thirty-Second AAAI Conference on Artificial
            Intelligence, 2018.
            http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit
        """
        rank_mat = torch.tensor(self.pair_rank_mat(idx_durations, events)).to(self.device)
        idx_durations = idx_durations.view(-1, 1)
        pmf = phi.softmax(1)
        y = torch.zeros_like(pmf).scatter(1, idx_durations, 1.) # one-hot
        rank_loss = self._rank_loss_deephit(pmf, y, rank_mat, self.sigma, reduction)
        return rank_loss
    

class DeepHitLoss(nn.Module):
    def __init__(self, sigma=0.1, weight=1, device='cuda'):
        '''
        Linear combination of NLL
        '''
        super().__init__()
        self.nll = NLLLoss(device=device)
        self.ranking = RankingLoss(sigma=sigma, device=device)
        self.weight = weight

    def forward(self, h, t, e):
        return self.nll(h, t, e) + self.weight * self.ranking(h, t, e)
    
