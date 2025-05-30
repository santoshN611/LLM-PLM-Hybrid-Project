
import torch
import torch.nn as nn
from transformers import BertModel

class ProteinAdapter(nn.Module):
    """
    Adapter block to inject into each Transformer layer:
      down-project -> act -> up-project -> add skip connection
    """
    def __init__(self, hidden_size, bottleneck=64):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.act  = nn.GELU()
        self.up   = nn.Linear(bottleneck, hidden_size)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


class BioBERTWithAdapters(nn.Module):
    def __init__(self, adapter_bottleneck=64):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            'dmis-lab/biobert-base-cased-v1.1',
            output_hidden_states=True
        )
        # freeze params
        for p in self.bert.parameters():
            p.requires_grad = False

        num_layers = self.bert.config.num_hidden_layers
        hidden_size = self.bert.config.hidden_size
        self.adapters = nn.ModuleList([
            ProteinAdapter(hidden_size, adapter_bottleneck)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states
        h = hidden_states[-1]

        
        for adapter in self.adapters:
            h = adapter(h)

        cls_emb = h[:, 0]
        logits  = self.classifier(cls_emb)
        return logits, cls_emb


def contrastive_loss(esm_emb: torch.Tensor, cls_emb: torch.Tensor, temperature=0.1):
    # align loss between bert and esm
    esm_norm = esm_emb / esm_emb.norm(dim=1, keepdim=True)
    cls_norm = cls_emb / cls_emb.norm(dim=1, keepdim=True)
    sim_matrix = torch.matmul(esm_norm, cls_norm.T) / temperature
    labels     = torch.arange(sim_matrix.size(0)).to(sim_matrix.device)
    return nn.CrossEntropyLoss()(sim_matrix, labels)