import copy
import gc
import torch
import torch.nn as nn
import numpy as np

from src.controllers.actions import (
    a0_NOOP, a1_BN_RECAL, a2_EM_PRIOR, a3_TENT_k
)

class HindsightOracle:
    """
    The Hindsight Oracle (HMO).
    Tests all actions by branching the model state, calculating future
    realized accuracy over the next H windows, and returning the optimal action.
    WARNING: Highly memory intensive. Aggressive cleanup required.
    """
    def __init__(self, actions: list[str], lambda_cost: float = 0.5):
        self.actions = actions
        self.lambda_cost = lambda_cost
        
    def evaluate_actions(
        self, 
        base_model: nn.Module, 
        buffer_x: torch.Tensor, 
        buffer_preds: torch.Tensor, 
        C_src: torch.Tensor,
        future_x: torch.Tensor, 
        future_y: torch.Tensor
    ) -> tuple[str, float]:
        """
        Evaluate all actions and return (best_action_name, max_utility).
        """
        device = next(base_model.parameters()).device
        base_state = copy.deepcopy(base_model.state_dict())
        
        best_action = "a0_NOOP"
        max_utility = -float('inf')
        
        # Calculate baseline accuracy (NOOP)
        # This acts as the anchor for delta_H
        base_model.eval()
        with torch.no_grad():
            base_logits = base_model(future_x)
            base_preds = base_logits.argmax(dim=1)
            base_acc = (base_preds == future_y).float().mean().item()
        
        for action_name in self.actions:
            # 1. Branch
            temp_model = copy.deepcopy(base_model)
            temp_model.load_state_dict(base_state)
            
            ops = 0
            # 2. Apply action
            if action_name == "a0_NOOP":
                temp_model, op = a0_NOOP(temp_model, buffer_x)
                ops += op
            elif action_name == "a1_BN_RECAL":
                temp_model, op = a1_BN_RECAL(temp_model, buffer_x, m=0.5)
                ops += op
            elif action_name == "a2_EM_PRIOR":
                temp_model, op = a2_EM_PRIOR(temp_model, buffer_preds, C_src)
                ops += op
            elif action_name == "a3_TENT_k":
                temp_model, op = a3_TENT_k(temp_model, buffer_x, k=2)
                ops += op
                
            # 3. Evaluate Future H (NOOP)
            temp_model.eval()
            with torch.no_grad():
                logits = temp_model(future_x)
                preds = logits.argmax(dim=1)
                acc = (preds == future_y).float().mean().item()
                
            delta_H = acc - base_acc
            
            # Simple scaling for op cost
            # In a real setup, ops are mapped to mJ via profiles.yaml
            # For the oracle selection mechanism here, we mock the penalty
            cost = (ops / 88800000.0) 
            utility = delta_H - (self.lambda_cost * cost)
            
            if utility > max_utility:
                max_utility = utility
                best_action = action_name
                
            # 4. NUKE (CRITICAL MEMORY MANAGEMENT)
            del temp_model
            del logits
            del preds
            
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
        # Clean up baseline tensors
        del base_state
        del base_logits
        del base_preds
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        return best_action, max_utility
