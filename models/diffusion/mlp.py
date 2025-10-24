import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiTAdaLayerNorm(nn.Module):
    def __init__(self, feature_dim, epsilon=1e-6):
        super(DiTAdaLayerNorm, self).__init__()
        self.epsilon = epsilon
        self.weight = torch.nn.Parameter(torch.randn(feature_dim, feature_dim * 2))

    def __call__(self, x, condition):
        """
        Args:
            x (np.ndarray): shape: (batch_size, feature_dim)
            condition (np.ndarray): shape: (batch_size, 1, feature_dim)
                Ps: condition = time_cond_embedding + class_cond_embedding
        return:
            x_layer_norm (np.ndarray): shape: (batch_size, sequence_length, feature_dim)
        """
        # print("[DEBUG] condition.shape: ", condition.shape)
        # print("[DEBUG] self.weight.shape: ", self.weight.shape)
        # print("[DEBUG] x.shape: ", x.shape)
        affine = condition @ self.weight  # shape: (batch_size, feature_dim * 2)
        # print("[DEBUG] affine.shape:", affine.shape)
        # print(affine)
        gamma, beta = torch.split(affine, affine.shape[-1] // 2, dim=-1)
        _mean = torch.mean(x, dim=-1, keepdims=True)
        _std = torch.var(x, dim=-1, keepdims=True)
        x_layer_norm = gamma * (x - _mean / (_std + self.epsilon)) + beta
        return x_layer_norm

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        raise NotImplementedError
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps])

class MLP_Denoiser(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim, output_dim, num_layers=3):
        super(MLP_Denoiser, self).__init__()

        self.num_layers = num_layers
        self.mlp_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()
        self.ln_forward_layers = nn.ModuleList()

        # build projector
        self.input_embedder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.condition_embedder = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        
        # build middle layers
        for _ in range(num_layers - 1):
            self.ln_layers.append(DiTAdaLayerNorm(hidden_dim))
            self.mlp_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
        
        # build last layer
        self.ln_layers.append(DiTAdaLayerNorm(hidden_dim))
        self.mlp_layers.append(nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        ))
    
    def forward(self, x, cond):
        x = self.input_embedder(x)
        cond = self.condition_embedder(cond)
        for i in range(self.num_layers):
            x = self.ln_layers[i](x, cond)
            x = self.mlp_layers[i](x)
        return x

class LDM(nn.Module):
    def __init__(self, modeltype, input_dim, num_actions, translation,
                 cond_dim=512, ff_size=1024, output_dim=512, num_layers=8, dropout=0,
                 ablation=None, legacy=False, arch='mlp', clip_version=None, **kargs):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.num_actions = num_actions

        self.translation = translation

        self.cond_dim = cond_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.ablation = ablation
        self.output_dim = output_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_dim = input_dim

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch

        self.sequence_pos_encoder = PositionalEncoding(self.cond_dim, self.dropout)

        self.mlp = MLP_Denoiser(self.input_dim, self.cond_dim, self.ff_size, self.output_dim, num_layers=self.num_layers)
        self.embed_timestep = TimestepEmbedder(self.cond_dim, self.sequence_pos_encoder)


    def forward(self, x, timesteps, y):
        """
        x: [batch_size, token_emb]
        timesteps: [batch_size] (int)
        """
        timestep_emb = self.embed_timestep(timesteps)
        cond = y['text_embed']
        # print("[DEBUG] cond.shape: ", cond.shape)
        # print("[DEBUG] timestep_emb.shape: ", timestep_emb.shape)
        xseq = x
        cond = cond + timestep_emb
        output = self.mlp(xseq, cond)

        return output
    