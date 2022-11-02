# model parameter setting
class Config():
    def __init__(self):
        self.n_layers = 6
        self.n_heads = 8
        self.d_model = 512
        self.attn_mask = None