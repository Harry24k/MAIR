import torch


@torch.no_grad()
def get_accuracy(model, data_loader, atk=None, n_limit=1e10, device=None):
    model = model.eval()

    if device is None:
        device = next(model.parameters()).device

    correct = 0
    total = 0

    for images, labels in data_loader:

        X = images.to(device)
        Y = labels.to(device)

        if atk:
            X = atk(X, Y)

        pre = model(X)

        _, pre = torch.max(pre.data, 1)
        total += pre.size(0)
        correct += (pre == Y).sum()

        if total > n_limit:
            break

    return 100 * float(correct) / total
