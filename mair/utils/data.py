def get_subloader(given_loader, n_limit):
    if n_limit is None:
        return given_loader

    sub_loader = []
    num = 0
    for item in given_loader:
        sub_loader.append(item)
        if isinstance(item, tuple) or isinstance(item, list):
            batch_size = len(item[0])
        else:
            batch_size = len(item)
        num += batch_size
        if num >= n_limit:
            break
    return sub_loader
