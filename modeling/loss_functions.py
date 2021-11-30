import torch
import torch.nn
from dataclasses import dataclass
from typing import Union, Optional
import math


def stable_softmax(input, dim):
    c = input.max(dim=dim, keepdim=True).values
    return torch.nn.functional.softmax(input - c, dim)


@dataclass
class ProbOverDocs():
    score: Optional[torch.tensor]
    prob: Optional[torch.tensor]
    log_prob: Optional[torch.tensor]

    @classmethod
    def from_score(cls, score):
        score = score.refine_names('instance', 'doc')
        prob = stable_softmax(score, dim='doc')
        log_prob = torch.nn.functional.log_softmax(score, dim='doc')
        return cls(score, prob, log_prob)

    @classmethod
    def from_prob(cls, prob):
        prob = prob.refine_names('instance', 'doc')
        eps = 1e-7
        log_prob = (prob+eps).log()
        return cls(None, prob, log_prob)


    @classmethod
    def only_log_prob(cls, log_prob):
        log_prob = log_prob.refine_names('instance', 'doc')
        prob = log_prob.exp()
        # Warning: Downstream usage of score and prob might throw errors because they would be None
        return cls(None, prob, log_prob)


def marg_loss(gen_prob: ProbOverDocs,
              ret_prob: ProbOverDocs) -> torch.Tensor:
    """
    Compute P(y|x) by marginalizing over documents z using retriever probabilities P_ret(z|x) and generator
    probabilities P_gen(y|x,z)

    Args:
        gen_prob: Probs from the generator P_gen(y|x,z); has shape (batch_size, n_docs);
        ret_prob: Probs from the retriever P_ret(y|x); has shape (batch_size, n_docs)

    Returns:
        Tensor with shape (batch_size), each value P(y|x)
    """
    loss = -torch.logsumexp(ret_prob.log_prob + gen_prob.log_prob, dim='doc').sum(dim='instance')
    return loss


def reconstruction_loss(gen_prob: ProbOverDocs, guide_prob: ProbOverDocs):
    """
    Compute $$\\sum_{z} Q(z|x,y)log(P(y|x,z))$$
    Args:
        guide_prob: Probs from the label-posterior retriever (or guide) Q(z|x,y); has shape (batch_size, n_docs)
        gen_prob: Probs from the generator P_gen(y|x,z); has shape (batch_size, n_docs);

    Returns:
        Tensor with shape (batch_size)
    """
    reconstruction_loss = -(guide_prob.prob * gen_prob.log_prob).sum()
    return reconstruction_loss


def kld(guide_prob: ProbOverDocs, ret_prob: ProbOverDocs) -> torch.Tensor:
    """
    Compute KL Divergence, D_KL(Q(z|x,y), P(z|x)) between label-posterior retriever (guide) and the actual retriever
    Args:
        guide_prob: Probs from the label-posterior retriever (or guide) Q(z|x,y); has shape (batch_size, n_docs)
        ret_prob: Probs from the retriever P_ret(y|x); has shape (batch_size, n_docs)

    Returns:
        D_KL(Q|P)
    """
    kld = (guide_prob.prob * (guide_prob.log_prob - ret_prob.log_prob)).sum()
    return kld


def elbo_loss(gen_prob: ProbOverDocs, ret_prob: ProbOverDocs, guide_prob: ProbOverDocs):
    return reconstruction_loss(gen_prob, guide_prob) + kld(guide_prob, ret_prob)


class TestLossFunctions:
    def test_marg_loss(self):
        gen_prob = ProbOverDocs.from_prob(torch.tensor([[.1, .2, .3], [0.4, 0.5, 0.6]]))
        ret_prob = ProbOverDocs.from_prob(torch.tensor([[.4, .3, .3], [0.5, 0.5, 0.0]]))
        assert (marg_loss(gen_prob, ret_prob) == -(math.log(0.19) + math.log(0.45)))

    def test_kld(self):
        guide_prob = ProbOverDocs.from_prob(torch.tensor([[.8, .1, .1], [0.9, 0.1, 0.0]]))
        ret_prob = ProbOverDocs.from_prob(torch.tensor([[.4, .3, .3], [0.5, 0.5, 0.0]]))
        assert (kld(guide_prob, ret_prob) == .8*(math.log(.8/.4)) + 0.1*math.log(.1/.3) + 0.1*math.log(.1/.3)
                                            +.9*math.log(.9/.5) + .1*math.log(.1/.5))

    def test_reconstruction_loss(self):
        gen_prob = ProbOverDocs.from_prob(torch.tensor([[.1, .2, .3], [0.4, 0.5, 0.6]]))
        guide_prob = ProbOverDocs.from_prob(torch.tensor([[.8, .1, .1], [0.9, 0.1, 0.0]]))
        assert(reconstruction_loss(gen_prob, guide_prob).isclose(torch.Tensor(
            [-(.8*math.log(.1) + .1*math.log(.2) + .1*math.log(.3)
             +.9*math.log(.4) + .1*math.log(.5))])))

    def test_elbo(self):
        gen_prob = ProbOverDocs.from_prob(torch.tensor([[.1, .2, .3], [0.4, 0.5, 0.6]]))
        guide_prob = ProbOverDocs.from_prob(torch.tensor([[.8, .1, .1], [0.9, 0.1, 0.0]]))
        ret_prob = ProbOverDocs.from_prob(torch.tensor([[.4, .3, .3], [0.5, 0.5, 0.0]]))
        assert elbo_loss(gen_prob, ret_prob, guide_prob).isclose(torch.Tensor(
            [.8*(math.log(.8/.4/.1)) + 0.1*math.log(.1/.3/.2) + 0.1*math.log(.1/.3/.3)
                                            +.9*math.log(.9/.5/.4) + .1*math.log(.1/.5/.5)]))


