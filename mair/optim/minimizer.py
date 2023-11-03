from collections import defaultdict

import torch


class Minimizer(object):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.state = defaultdict(dict)
        self.records = {}

    def update_records(self, key, value):
        arr = self.records.get(key)
        if arr is None:
            self.records[key] = []
            arr = self.records.get(key)
        arr.append(value)

    def state_dict(self):
        """Returns the state of the minimizer as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ["model", "optimizer", "state"]
        }

    def load_state_dict(self, state_dict):
        """Loads the minimizers state.
        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def step(self):
        raise NotImplementedError


class SAM(Minimizer):
    def __init__(self, model, optimizer, rho):
        super().__init__(model, optimizer)
        self.rho = rho

    def step(self, cost_fn, *inputs):
        cost = cost_fn(*inputs)
        cost.backward()
        self.ascent_step()

        self.update_records("loss_0", cost.item())

        cost = cost_fn(*inputs)
        cost.backward()
        self.descent_step()

        self.update_records("loss_p", cost.item())

    @torch.no_grad()
    def ascent_step(self):
        grad_norm = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grad_norm.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grad_norm), p=2) + 1.0e-16

        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            self.state[p]["eps"] = self.rho / grad_norm * p.grad.clone().detach()
            p.add_(self.state[p]["eps"])
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()


class ASAM(SAM):
    def __init__(self, optimizer, model, rho, eta=0.01):
        super().__init__(optimizer, model, rho)
        self.eta = eta

    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if "weight" in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.0e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if "weight" in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()


class GSAM(SAM):
    def __init__(
        self,
        optimizer,
        model,
        rho,
        alpha=0.1,
        decay=True,
        rho_min=0,
        lr_max=None,
        lr_min=0,
    ):
        super().__init__(optimizer, model, rho)
        self.alpha = alpha
        self.decay = decay
        assert self.decay and (lr_max is not None)
        self.rho_min = rho_min
        self.lr_max = lr_max
        self.lr_min = lr_min

    @torch.no_grad()
    def ascent_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            self.state["ascent"][n] = p.grad.clone().detach()
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.0e-16
        self.ascent_norm = grad_norm.clone().detach()

        if self.decay:
            lr = self.optimizer.param_groups[0]["lr"]
            rho = self.rho_min + (self.rho - self.rho_min) * (lr - self.lr_min) / (
                self.lr_max - self.lr_min
            )
        else:
            rho = self.rho

        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        self.state["descent"] = {}
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            self.state["descent"][n] = p.grad.clone().detach()
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.0e-16
        self.descent_norm = grad_norm

        descents = self.state["descent"]
        ascents = self.state["ascent"]

        inner_prod = self.inner_product(descents, ascents)

        # get cosine
        cosine = inner_prod / (self.ascent_norm * self.descent_norm + 1.0e-16)

        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            vertical = ascents[n] - cosine * self.ascent_norm * descents[n] / (
                self.descent_norm + 1.0e-16
            )
            p.grad.sub_(self.alpha * vertical)

        self.optimizer.step()
        self.optimizer.zero_grad()

    def inner_product(self, u, v):
        value = 0
        for key in v.keys():
            value += torch.dot(u[key].flatten(), v[key].flatten())
        return value


class BridgedNSAM(SAM):
    def __init__(self, optimizer, model, rho, N=None, rhos=None, descents_weights=None):
        super().__init__(optimizer, model, rho)
        assert (N is not None) or (rhos is not None)
        if N is not None:
            self.N = N
            self.rhos = [rho / N] * N
        if rhos is not None:
            self.rhos = rhos
            self.N = len(rhos)
        if descents_weights is None:
            self.descents_weights = [1 / N] * N

    def step(self, cost_fn, *inputs):
        self.n = 0
        self.rho = self.rhos[self.n]
        cost = cost_fn(*inputs)
        cost.backward()
        self.ascent_step()

        self.update_records("loss_0", cost.item())

        for n in range(self.N - 2):
            self.n = n
            self.rho = self.rhos[self.n]
            cost = cost_fn(*inputs)
            cost.backward()
            self.middle_step()

            self.update_records("loss_%d" % n, cost.item())

        self.n = n + 1
        self.rho = self.rhos[self.n]
        cost = cost_fn(*inputs)
        cost.backward()
        self.descent_step()

        self.update_records("loss_p", cost.item())

    # Calculate First Grad and First Grad Norm. Second Ascent by First Grad.
    @torch.no_grad()
    def middle_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            self.state[p]["grad"] = (
                self.descents_weights[self.n] * p.grad.clone().detach()
            )  # Add grad
            grads.append(
                torch.norm(p.grad.clone().detach(), p=2)
            )  # Current grad to calculate norm
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.0e-16

        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps_t = self.rho * p.grad.clone().detach() / grad_norm
            self.state[p]["eps"] += eps_t  # Accumulate eps
            p.add_(eps_t)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])  # Back ALL eps
            p.grad.mul_(self.descents_weights[self.n])
            p.grad.add_(self.state[p]["grad"])  # Add ALL grads

        self.optimizer.step()
        self.optimizer.zero_grad()


class AWP(SAM):
    @torch.no_grad()
    def ascent_step(self):
        weights = []
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            weights.append(torch.norm(p, p=2))
            grads.append(torch.norm(p.grad, p=2))
        weight_norm = torch.norm(torch.stack(weights), p=2) + 1.0e-16
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.0e-16

        self.update_records("weight_norm", weight_norm.item())
        self.update_records("grad_norm", grad_norm.item())

        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho * weight_norm / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()
