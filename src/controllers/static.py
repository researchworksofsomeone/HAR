import torch
import torch.nn as nn

from src.controllers.actions import (
    a0_NOOP, a1_BN_RECAL, a2_EM_PRIOR, a3_TENT_k, a4_RESET
)

class StaticController:
    """Unified static baseline controller."""
    def __init__(self, policy: str):
        valid_policies = ["SRC", "TENT_ALWAYS", "BN_ALWAYS", "EM_ALWAYS", "CASCADE", "PERIODIC_TENT"]
        assert policy in valid_policies, f"Invalid policy: {policy}"
        self.policy = policy
        self.tick_count = 0
        
    def step(self, model: nn.Module, buffer_x: torch.Tensor, buffer_preds: torch.Tensor, C_src: torch.Tensor = None) -> tuple[nn.Module, int]:
        """
        Executes the static policy on the given model and buffers.
        Returns:
            updated_model, ops
        """
        self.tick_count += 1
        ops = 0
        
        if self.policy == "SRC":
            model, op = a0_NOOP(model, buffer_x)
            ops += op
            
        elif self.policy == "TENT_ALWAYS":
            model, op = a3_TENT_k(model, buffer_x, k=2)
            ops += op
            
        elif self.policy == "BN_ALWAYS":
            model, op = a1_BN_RECAL(model, buffer_x, m=0.5)
            ops += op
            
        elif self.policy == "EM_ALWAYS":
            model, op = a2_EM_PRIOR(model, buffer_preds, C_src)
            ops += op
            
        elif self.policy == "CASCADE":
            model, op1 = a1_BN_RECAL(model, buffer_x, m=0.5)
            model, op2 = a2_EM_PRIOR(model, buffer_preds, C_src)
            model, op3 = a3_TENT_k(model, buffer_x, k=2)
            ops += op1 + op2 + op3
            
        elif self.policy == "PERIODIC_TENT":
            if self.tick_count % 8 == 0:
                model, op = a3_TENT_k(model, buffer_x, k=2)
                ops += op
            else:
                model, op = a0_NOOP(model, buffer_x)
                ops += op
                
        return model, ops
