from typing import List

def get_masked_actions(profile: str, available_ram_mb: int) -> List[int]:
    """
    Returns the allowed actions based on the device profile and current available RAM.
    Actions mapping:
    0: a0_NOOP (0 RAM, 0 FLOPs)
    1: a1_BN_RECAL (Tiny RAM overhead)
    2: a2_EM_PRIOR (Low RAM, logits only)
    3: a3_TENT_k (Gradient-based, highest RAM)
    4: a4_LITE (Pseudo-action for degraded TENT)
    
    Rules (from proposal §5.5):
    - P1 (Cortex-M4F, 2MB): No mask (all actions allowed).
    - P2 (Cortex-M7, 512KB): Mask a3_TENT_k unless buffer fits; else fallback to a4_LITE.
    - P3 (Cortex-M4, 256KB): Mask all gradient-based actions. Allowed: {0, 1, 2, 4}.
    """
    all_actions = [0, 1, 2, 3] # assuming 4 base actions for now. 4=lite can be added.
    
    if profile == "P1":
        # Cortex-M4F / 2MB / No constraints
        return [0, 1, 2, 3]
        
    elif profile == "P2":
        # Cortex-M7 / 512KB
        # If RAM is critical (< 0.2 MB buffer left), fallback to lighter actions
        if available_ram_mb < 0.2:
            return [0, 1, 2] # Drop TENT
        else:
            return [0, 1, 2, 3]
            
    elif profile == "P3":
        # Cortex-M4 / 256KB
        # No gradient-based actions allowed EVER
        return [0, 1, 2]
        
    elif profile == "P4":
        # Cortex-M0+ / 128KB (Extreme)
        # Maybe only NOOP and BN_RECAL
        return [0, 1]
        
    # Default fallback: allow all safely
    return [0, 1, 2, 3]
