from torch import nn,Tensor
from typing import Optional
from torch.nn import functional as F

class VTRM(nn.Module):
    def __init__(self, d_model, nhead, visual_dim=1024, text_dim=768):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, text_dim)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(77)
        self.norm = nn.LayerNorm(d_model)
        self.activation = F.relu
        self._reset_parameters()
        self.MLP = nn.Linear(d_model, d_model)
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    def forward(self, tgt, memory, flat:str,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,):
        
        # memory : h*w,b,c     tgt : l,b,d
        memory = self.visual_proj(memory)
        memory = self.adaptive_pool(memory.permute(1,2,0))  # [8, 768, 77]
        memory = memory.permute(2, 0, 1)
        if flat=="text": 
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos), # text
                                    key=self.with_pos_embed(memory, pos), # vision
                                    value=memory, attn_mask=memory_mask, # vision
                                    key_padding_mask=memory_key_padding_mask)[0]
        elif flat=="vision":
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, pos), # vision
                        key=self.with_pos_embed(memory, query_pos), # text
                        value=memory, attn_mask=memory_mask, # text
                        key_padding_mask=memory_key_padding_mask)[0]

        tgt = tgt + self.MLP(tgt2)
        tgt = self.norm(tgt)
        return tgt
