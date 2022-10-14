import normflows as nf
import torch
import torch.nn as nn
from normflows.flows import Planar, Radial


class Flow(nn.Module):
    def __init__(self, latent_size, flow, K):
        super().__init__()
        if flow == "Planar":
            flows = [Planar((latent_size,)) for _ in range(K)]
        elif flow == "Radial":
            flows = [Radial((latent_size,)) for _ in range(K)]
        elif flow == "RealNVP":
            flows = []
            b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
            for i in range(K):
                s = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
                t = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
                if i % 2 == 0:
                    flows += [nf.flows.MaskedAffineFlow(b, t, s)]
                else:
                    flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
        self.flows = nn.ModuleList(flows)

    def forward(self, z, base_dist, prior=None):
        ld = 0.0
        p_0 = torch.sum(base_dist.log_prob(z), -1)
        for flow in self.flows:
            z, ld_ = flow(z)
            ld += ld_
        # z = z.squeeze_()
        # KLD including logdet term
        if prior:
            kld = p_0 - torch.sum(prior.log_prob(z), -1) - ld.view(-1)
        else:
            kld = None

        return z, kld
