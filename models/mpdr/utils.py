import torch


def orthogonalize_batch(v, p):
    '''
    project v onto the orthogonal plane of p
    v: vector to be orthogonalize (BxD)
    p: unit vector (BxD)
    '''
    p_ = p.unsqueeze(1)  # (Bx1xD)
    v_ = v.unsqueeze(2)  # (BxDx1)
    pp = p_.transpose(1,2).matmul(p_)  # (BxDxD)
    projected = torch.matmul(pp, v_) # (BxDx1)
    return v - projected.squeeze(2)  # (BxD)


def orthogonalize_batch_v2(v, p):
    """
    project v onto the orthogonal plane of p
    v: vector to be orthogonalize (BxDx...)
    p: unit vector (BxDx...)
    """
    return v - (v*p).sum(dim=1, keepdim=True)*p



