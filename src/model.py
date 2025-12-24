import torch as pt
from torch import nn


def get_topk_mask(x, k, dim=-1):
    _, indices = pt.topk(x, k, dim=dim, sorted=False)
    mask = pt.full_like(x, -pt.inf)
    mask.scatter_(dim, indices, 0)
    
    return mask


class CoLa(nn.Module):
    def __init__(self, in_features, out_features, sub_features, q_features, num_subnets, num_active):
        super().__init__()
        self.k_layer = nn.Linear(q_features, num_subnets)
        self.V_0 = nn.Parameter(pt.randn(num_subnets, sub_features, in_features))
        self.V_1 = nn.Parameter(pt.randn(num_subnets, out_features, sub_features))
        self.softmax = nn.Softmax(dim=1)
        self.num_subnets = num_subnets
        self.num_active = num_active

    def forward(self, x, q):
        attention = self.k_layer(q)
        attention_mask = get_topk_mask(attention, self.num_active)
        sparse_attention = self.softmax(attention + attention_mask)
        
        V = pt.bmm(self.V_1, self.V_0)
        
        # Aggregate weights: (Batch, Subnets) x (Subnets, Out, In) -> (Batch, Out, In)
        weights = pt.einsum('bs,soi->boi', sparse_attention, V)
        
        # Apply weights: (Batch, Out, In) x (Batch, In) -> (Batch, Out)
        return pt.einsum('boi,bi->bo', weights, x)


class TaskLinear(nn.Module):
    def __init__(self, in_features, out_features, num_tasks):
        super().__init__()
        self.weights = nn.Parameter(pt.randn(num_tasks, out_features, in_features))
        self.bias = nn.Parameter(pt.randn(num_tasks, out_features))

    def forward(self, x, task_id):
        w = self.weights[task_id]
        b = self.bias[task_id]
        return pt.einsum('boi,bi->bo', w, x) + b
    

class MTasker(nn.Module):
    def __init__(self, in_features, out_features, enc_features, dec_features, num_tasks, sub_features, q_features, num_subnets, num_active):
        super().__init__()
        self.encoder = TaskLinear(in_features, enc_features, num_tasks)
        self.decoder = TaskLinear(dec_features, out_features, num_tasks)
        
        self.qs = nn.Parameter(pt.randn(num_tasks, q_features))
        self.cola = CoLa(enc_features, dec_features, sub_features, q_features, num_subnets, num_active)

        self.relu = nn.ReLU()
    
    def forward(self, x, task_ids):
        x = self.encoder(x, task_ids)
        x = self.relu(x)

        q = self.qs[task_ids]
        x = self.cola(x, q)

        x = self.relu(x)
        x = self.decoder(x, task_ids)
        return x


if __name__ == "__main__":
    B = 4
    IN_F = 16
    OUT_F = 8
    ENC_F = 12
    DEC_F = 14
    SUB_F = 6
    Q_F = 2
    N_SUB = 10
    N_ACT = 5
    N_TASKS = 7

    print("\nTesting CoLa")
    cola = CoLa(IN_F, OUT_F, SUB_F, Q_F, N_SUB, N_ACT)
    x = pt.randn(B, IN_F)
    q = pt.randn(B, Q_F)
    print(f"Input shape: {x.shape}, Q shape: {q.shape}")
    print(f"Output shape: {cola(x, q).shape}")

    print("\nTesting TaskLinear")
    tl = TaskLinear(IN_F, OUT_F, N_TASKS)
    task_ids = pt.randint(0, N_TASKS, (B,))
    print(f"Input shape: {x.shape}, Task IDs shape: {task_ids.shape}")
    print(f"Output shape: {tl(x, task_ids).shape}")

    print("\nTesting MTasker")
    mt = MTasker(IN_F, OUT_F, ENC_F, DEC_F, N_TASKS, SUB_F, Q_F, N_SUB, N_ACT)
    print(f"Input shape: {x.shape}, Task IDs shape: {task_ids.shape}")
    print(f"Output shape: {mt(x, task_ids).shape}")
