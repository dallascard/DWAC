

def to_numpy(var, device):
    if device != 'cpu':
        return var.cpu().data.numpy()
    else:
        return var.data.numpy()

