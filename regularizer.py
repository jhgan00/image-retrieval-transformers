import torch


class DifferentialEntropyRegularization(torch.nn.Module):

    def __init__(self, eps=1e-8):
        super(DifferentialEntropyRegularization, self).__init__()
        self.eps = eps
        self.pdist = torch.nn.PairwiseDistance(2)

    def forward(self, x):

        with torch.no_grad():
            dots = torch.mm(x, x.t())
            n = x.shape[0]
            dots.view(-1)[::(n + 1)].fill_(-1)  # trick to fill diagonal with -1
            _, I = torch.max(dots, 1)  # max inner prod -> min distance

        rho = self.pdist(x, x[I])

        # dist_matrix = torch.norm(x.unsqueeze(1) - x.unsqueeze(0), p=2, dim=-1)
        # rho = dist_matrix.topk(k=2, largest=False)[0][:, 1]

        loss = -torch.log(rho + self.eps).mean()

        return loss

