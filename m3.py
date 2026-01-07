import torch
import torch.distributed as dist

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz orthogonalization for the Muon/M3 update.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # Ensure Frobenius norm <= 1
    if G.size(0) > G.size(1):
        X = X.T
    
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

class M3Optimizer(torch.optim.Optimizer):
    """
    Multi-scale Momentum Muon (M3) Optimizer.
    Adapts 'Nested Learning' by using two momentum buffers:
    - Fast Memory: Updates every step (standard momentum).
    - Slow Memory: Updates every `slow_freq` steps (long-term memory).
    """
    def __init__(self, params, lr=0.02, momentum=0.95, slow_momentum=0.99, 
                 slow_freq=10, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, slow_momentum=slow_momentum, 
                        slow_freq=slow_freq, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            slow_momentum = group['slow_momentum']
            slow_freq = group['slow_freq']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                # Initialize State
                if len(state) == 0:
                    state['step'] = 0
                    state['fast_buffer'] = torch.zeros_like(p.data)
                    state['slow_buffer'] = torch.zeros_like(p.data)

                state['step'] += 1
                fast_buf = state['fast_buffer']
                slow_buf = state['slow_buffer']

                # 1. Update Fast Memory (Standard Momentum)
                fast_buf.mul_(momentum).add_(grad)

                # 2. Update Slow Memory (Nested Level)
                if state['step'] % slow_freq == 0:
                    slow_buf.mul_(slow_momentum).add_(fast_buf)

                # 3. Combine for Update (Nested Learning Injection)
                # We use the fast buffer primarily, modulated by the slow buffer structure
                update_tensor = fast_buf + (0.5 * slow_buf)

                # 4. Apply Newton-Schulz (Muon adaptation)
                # Only apply to 2D matrices (linear layers); vector params use standard SGD
                if update_tensor.ndim == 2:
                    ortho_update = zeropower_via_newtonschulz5(update_tensor, steps=ns_steps)
                    p.data.add_(ortho_update, alpha=-lr)
                else:
                    p.data.add_(update_tensor, alpha=-lr)