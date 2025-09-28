import torch.nn as nn
import random
from collections import deque

class Adapter(nn.Module):
    """
    Simple bottleneck adapter: down-project -> nonlinearity -> up-project with residual.
    Applies per-token (same across time).
    """
    def __init__(self, d_model, bottleneck):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck)
        self.act = nn.ReLU()
        self.up = nn.Linear(bottleneck, d_model)

    def forward(self, x):
        # x: B x T x D
        z = self.down(x)     # B T b
        z = self.act(z)
        z = self.up(z)       # B T D
        return x + z         # residual

def attach_adapters(model, bottleneck=64, n_adapters=1):
    """
    Attach adapters to a TemporalTransformer-like model.
    - Creates `n_adapters` Adapter modules and assigns them as model.adapters (nn.ModuleList).
    - Returns the modified model.
    """
    # infer d_model if possible
    d_model = None
    if hasattr(model, 'input_proj') and hasattr(model.input_proj, 'out_features'):
        d_model = model.input_proj.out_features
    # fallback
    if d_model is None:
        # try attribute
        d_model = getattr(model, 'd_model', 256)

    adapters = nn.ModuleList([Adapter(d_model, bottleneck) for _ in range(n_adapters)])
    model.adapters = adapters
    return model

# --------------- Replay buffers (duplicate-safe, same semantics) ----------------
class ReservoirBuffer:
    """Reservoir sampling buffer for streaming samples."""
    def __init__(self, capacity, seed=42):
        self.capacity = capacity
        self.buffer = []
        self.n_seen = 0
        random.seed(seed)

    def add(self, sample):
        # sample: (emb_numpy, label, vid_id)
        self.n_seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            i = random.randint(0, self.n_seen-1)
            if i < self.capacity:
                self.buffer[i] = sample

    def sample(self, k):
        if len(self.buffer) == 0:
            return []
        k = min(k, len(self.buffer))
        return random.sample(self.buffer, k)

class ClassBalancedBuffer:
    """DMP-style: keep per-class queues with quotas."""
    def __init__(self, capacity, classes=[0,1]):
        self.capacity = capacity
        self.per_class = {c: deque(maxlen=max(1, capacity//len(classes))) for c in classes}
        self.classes = classes

    def add(self, sample):
        emb_np, lbl, vid = sample
        c = int(lbl)
        if c not in self.per_class:
            return
        self.per_class[c].append(sample)

    def sample(self, k):
        buffers = []
        for c in self.classes:
            buffers.extend(list(self.per_class[c]))
        if not buffers:
            return []
        k = min(k, len(buffers))
        return random.sample(buffers, k)
