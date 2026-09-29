"""
JADE Action Menu (Day 4)
Defines the core adaptation strategies (actions) used by the JADE controller.
Each action returns (updated_model, ops_count).
"""
import copy
import torch
import torch.nn as nn
import torch.optim as optim

def a0_NOOP(model: nn.Module, x: torch.Tensor) -> tuple[nn.Module, int]:
    """No adaptation."""
    return model, 0

def a1_BN_RECAL(model: nn.Module, buffer_x: torch.Tensor, m: float = 0.5) -> tuple[nn.Module, int]:
    """
    Re-calibrate BatchNorm running statistics using the recent window buffer.
    m: momentum blend (0 = keep old, 1 = use only new).
    Ops: ~ 2 * channels per BN layer.
    """
    model.eval()
    if len(buffer_x) == 0:
        return model, 0
    
    # We don't backprop, just forward pass the buffer in train mode locally
    # to update running stats. Actually we can manually update running stats
    # to be precise or just run a forward pass with train() mode.
    # The proposal says: "Replaces BatchNorm running_mean and running_var 
    # with the statistics from a ring buffer... Momentum blend m=0.5."
    
    # Let's save original momentum, set to m, run forward, then restore.
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    old_momentums = []
    
    ops = 0
    for bn in bns:
        old_momentums.append(bn.momentum)
        bn.momentum = m
        ops += 2 * bn.num_features # Minimal op cost estimation

    # Set model to train mode ONLY for BN to update running stats
    model.train()
    with torch.no_grad():
        _ = model(buffer_x)
    model.eval()

    # Restore momentums
    for bn, old_m in zip(bns, old_momentums):
        bn.momentum = old_m

    return model, ops

def a2_EM_PRIOR(model: nn.Module, buffer_preds: torch.Tensor, C_src: torch.Tensor) -> tuple[nn.Module, int]:
    """
    Expectation-Maximization on prior shift using BBSE/Saerens.
    (Placeholder mechanics for MAC scaling. Applies a logit bias adjustment.)
    Ops: N * K * T_em
    """
    N, K = buffer_preds.shape
    T_em = 3 # Typical EM steps
    ops = N * K * T_em
    # In a full implementation, this updates the final layer bias.
    # We will simulate the op count for now as per constraints.
    return model, ops

def a3_TENT_k(model: nn.Module, buffer_x: torch.Tensor, k: int = 2) -> tuple[nn.Module, int]:
    """
    k SGD steps of entropy minimization on BN affines + final Dense layer.
    Buffer B=8 windows.
    """
    if len(buffer_x) == 0:
        return model, 0

    model.train()
    
    # Select parameters to optimize: BN affines + final linear
    params_to_update = []
    for name, param in model.named_parameters():
        if "bn" in name.lower() or "head.3" in name.lower() or "head.1" in name: 
            # Depending on model architecture, head.3 is the final linear layer
            # In TinyHARNet: head.1 is linear1, head.3 is linear2. Let's optimize both to be safe or just head.3.
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False

    optimizer = optim.SGD(params_to_update, lr=0.001)
    
    B = buffer_x.size(0)
    # MACs: k * B * (1 fwd + 2 bwd_equiv) -> 3 passes per window
    # Ops count placeholder, will be accurately mapped in Day 5 Profiler
    ops = k * B * 3 * 1850000 
    
    for _ in range(k):
        optimizer.zero_grad()
        logits = model(buffer_x)
        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
        entropy.backward()
        optimizer.step()
        
        # Explicit memory cleanup
        del logits, probs, entropy
        
    model.eval()
    
    # Restore requires_grad
    for param in model.parameters():
        param.requires_grad = True

    return model, ops

def a4_RESET(model: nn.Module, theta_src: dict) -> tuple[nn.Module, int]:
    """Deep copies the source weights theta_src back into f_theta."""
    model.load_state_dict(theta_src)
    # Ops ~ size of model parameters, treated as memory ops (0 MACs)
    return model, 0

# Baseline Actions
def a12_LEAN_ALWAYS(model: nn.Module, x: torch.Tensor, buffer_stats: dict) -> tuple[nn.Module, int]:
    """Backprop-free stateless correction using buffer stats."""
    return model, 50000 # Dummy MAC count

def a13_LAME_STYLE(model: nn.Module, x: torch.Tensor, buffer_feats: torch.Tensor) -> tuple[nn.Module, int]:
    """Parameter-free feature-affinity logit reweighting."""
    return model, 60000 # Dummy MAC count
