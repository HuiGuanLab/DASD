from collections import OrderedDict
import clip
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import transformers

m_toker = transformers.AutoTokenizer.from_pretrained('pretrained_model/bert-base-multilingual-cased')
     
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    

class Acquirer(nn.Module):
    def __init__(self, d_model, d_hidden, skip=False):
        super().__init__()
        
        self.skip = skip
        if not self.skip:

            self.mlp_sr = nn.Sequential(
                    OrderedDict([
                    ("c_fc", nn.Linear(d_model, d_hidden)),
                    ("relu", nn.ReLU()),
                    ("c_proj", nn.Linear(d_hidden, d_model))
                ])
            )

            self.mlp_sa = nn.Sequential(
                    OrderedDict([
                    ("c_fc", nn.Linear(d_model, d_hidden)),
                    ("relu", nn.ReLU()),
                    ("c_proj", nn.Linear(d_hidden, d_model))
                ])
            )
            self.mlp_beforez = nn.Sequential(
                    OrderedDict([
                    ("c_fc", nn.Linear(d_model, 32)),
                ])
            )
            self.mlp_afterz = nn.Sequential(
                    OrderedDict([
                    ("relu", nn.ReLU()),
                    ("c_proj2", nn.Linear(32, d_model))
                ])
            )
            self.ln = LayerNorm(d_model)
    
    def forward(self, x, zi = None, zi_bool = None):
        if self.skip:
            return x
        else:
            if zi_bool == None:
                output_specific = self.mlp_beforez(x)
                output_specific = torch.matmul(output_specific.permute(1,0,2), zi)
                output_specific = self.mlp_afterz(output_specific.permute(1,0,2))
                output = x + output_specific 

            elif zi_bool == 'sr':   
                sr = self.mlp_sr(x)
                output = x + sr
                
            elif zi_bool=='sa':
                sa = self.mlp_sa(x)
                output = x + sa  
        return output
    

class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        # self.config = config
        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = 0.1

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:

            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, acquirer=False, d_acquirer_hidden=256, m_acquirer=False, langs=None, skip=False):
    # def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        if acquirer:
            self.mlp_zi= nn.Sequential(OrderedDict([
                ("c_fc1", nn.Linear(512, 1024)),
            ]))

        self.acquirer = acquirer
        self.m_acquirer = m_acquirer
        if acquirer:
            if m_acquirer:
                assert isinstance(langs, list)
                self.acquirer = nn.ModuleDict(
                    {
                        lang: Acquirer(d_model, d_acquirer_hidden, skip=skip) for lang in langs
                    }
                )
            else:
                self.acquirer = Acquirer(d_model, d_acquirer_hidden, skip=skip)

    def attention(self, x: torch.Tensor, context_length, acquirer,ifviusal=False):

    
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None

        if ifviusal:
            return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)[1]
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, acquirer=False, lang=None, context_length=77,latent_semantics=None, zi_bool=None,acquirer_bool=True, mu=None,sigma=None,text_visual=None):
        orgin_data = x
        x = x + self.attention(self.ln_1(x), context_length=context_length, acquirer=acquirer)
        x = x + self.mlp(self.ln_2(x))

        if acquirer_bool == False:
            return x
        if acquirer:
            if acquirer_bool == False:
                return x
            if self.m_acquirer:
                if zi_bool == None:
                    zi = latent_semantics
                    zi = self.mlp_zi(zi)
                    zi = zi.reshape(-1,32,32)
                    x  = self.acquirer[lang](x,zi=zi,zi_bool=zi_bool)
            else:
                x = self.acquirer[lang](x, zi_bool=zi_bool)
        return x
    
    def build_attention_mask(self,context_length=77):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal

        return mask


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, acquirer=False, d_acquirer_hidden=256, m_acquirer=False, langs=None, skip=False):
    # def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, acquirer=False, d_acquirer_hidden=256):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, acquirer, d_acquirer_hidden, m_acquirer=m_acquirer, langs=langs, skip=skip) for _ in range(layers)])
        self.zi_adapter_student= nn.Sequential(OrderedDict([
                ("c_fc1", nn.Linear(1024, 512)),
                ("gelu1", QuickGELU()),
            ]))    
    def forward(self, x: torch.Tensor, context_length=77, return_all_layers=False, acquirer=False, lang=None,zi_bool=None,sr=None, sa=None,latent_semantics=None,inputlayers=12,text_visual=None):
        hiddens = [x]
        for index, layer in enumerate(self.resblocks):
            if index > inputlayers:
                return x
            if acquirer:
                if zi_bool == None:
                    local_info = torch.cat((sr, sa),dim=-1)
                    latent_semantics = self.zi_adapter_student(local_info)
                x = layer(x, acquirer, lang, latent_semantics = latent_semantics, zi_bool=zi_bool,text_visual = text_visual)
            else:
                x = layer(x, acquirer, lang, zi_bool=zi_bool,text_visual = text_visual)
            hiddens.append(x)
        if return_all_layers:
            return hiddens
        else:
            return x



class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 acquirer=False,
                 d_acquirer_hidden=256,
                 m_acquirer=False, 
                 langs=None,
                 skip=False,
                 init_mbert_embedding=True,
                 stage='CLA'
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            acquirer=acquirer,
            d_acquirer_hidden=d_acquirer_hidden,
            m_acquirer=m_acquirer, 
            langs=langs,
            skip=skip
        )   
        

        self.acquirer = acquirer
        self.init_mbert_embedding = init_mbert_embedding

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.dropout = nn.Dropout(0.2)
        self.ln_final = LayerNorm(transformer_width)
        self.ln_mu = LayerNorm(transformer_width)
        self.ln_sigma = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.target_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.mu_projection = nn.Parameter(torch.empty(transformer_width, embed_dim)) #TODO
        self.sigma_projection = nn.Parameter(torch.empty(transformer_width, embed_dim)) #TODO
        self.stage =stage

        from adv import Adversarial
        kwargs = {'input_size': 512 * 2,
                  'train_level': 'sent', 'train_type': 'GAN',
                  'reverse_grad': False, 'nclass': 2, 'scale': 0.001,
                  'optim': 'adam', 'lr': 1e-4, 'betas': (0.9, 0.999), 'gamma': 0, 'eps': 1e-8,
                  'momentum': 0.8, 'disc_type': 'not-so-weak'}
        self.AdvAgent = Adversarial(**kwargs)
        if self.acquirer:
            print('Using multilingual embedding...')
            self.multilingual_embedding = nn.Embedding(119547, 768)
            self.multilingual_embedding_linear = nn.Linear(768, 512)

            # self.multilingual_embedding_musigma = nn.Embedding(119547, 768)
            # self.multilingual_embedding_linear_mu = nn.Linear(768, 512)
            # self.multilingual_embedding_linear_sigma = nn.Linear(768, 512)
        self.initialize_parameters()


    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.acquirer:
            nn.init.normal_(self.multilingual_embedding.weight, std=0.02)
            if self.init_mbert_embedding:
                mbert_ckpt = torch.load('pretrained_model/bert-base-multilingual-cased/pytorch_model.bin')
                print('load mbert pretrained word embedding...')
                self.multilingual_embedding.weight.data = mbert_ckpt['bert.embeddings.word_embeddings.weight']
            else:
                print('do not use mbert init embeddings')

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
            # if self.acquirer:
            #     # nn.init.normal_(block.acquirer.nlp.linear.weight, std=proj_std)
            #     nn.init.normal_(block.acquirer.mlp.c_fc.weight, std=fc_std)
            #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)


    def get_input_embeddings(self):
        if not self.acquirer:
            return self.token_embedding
        else:
            return self.multilingual_embedding
        
    # #2023-11-2    
    def set_input_embeddings(self, value):
        if not self.acquirer:
            self.token_embedding = value
        else:
            self.multilingual_embedding = value    


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        
        return self.visual(image.type(self.dtype))

    
    def attn_embed(self, x):
        # import pdb;pdb.set_trace()
        # x = torch.matmul(x, self.token_embedding.weight.t()).softmax(dim=-1) * 
        x = torch.matmul(x, self.token_embedding.weight.t()).softmax(dim=-1) @ self.token_embedding.weight
        return x

    def encode_text(self, text, layers=None, acquirer=False, tokenize=False, mul_toker=None, lang=None, device='cuda',sr=None,sa=None,zi_bool=False,src_feats=None,istrain=False):

        context_length = self.context_length
        text_visual = text
        if tokenize:
            text = list(text)
            if acquirer:
                # import pdb;pdb.set_trace()
                if mul_toker:
                    text = mul_toker(text, truncate=True, padding='max_length').to(device)
                else:
                    text = m_toker(text, padding='max_length', max_length=77, truncation=True, return_tensors='pt')['input_ids'].to(device)          
                
                x_emb = self.multilingual_embedding(text).type(self.dtype)
                x = self.multilingual_embedding_linear(x_emb)

                x_musigma_emb = self.multilingual_embedding(text).type(self.dtype)
                x_mu = self.multilingual_embedding_linear(x_musigma_emb)
                x_sigma = self.multilingual_embedding_linear(x_musigma_emb)

            else:
                text = clip.tokenize(text, truncate=True).to(device)
                x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        else:
            x = self.token_embedding(text).type(self.dtype)  
        x = x + self.positional_embedding.type(self.dtype)
        
        if acquirer:
            x_mu = x_mu + self.positional_embedding.type(self.dtype)
            x_sigma = x_sigma + self.positional_embedding.type(self.dtype)
        if layers == None:
            if not zi_bool:
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.transformer(x, acquirer=acquirer, lang=lang, context_length=context_length,sr=sr, sa=sa , text_visual = text_visual)
                x = x.permute(1, 0, 2)  # LND -> NLD
                x = self.ln_final(x).type(self.dtype)
                if acquirer and tokenize and (mul_toker is None):
                    x = x[torch.arange(x.shape[0]), (text==102).nonzero()[:,1]] @ self.text_projection
                else:
                    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
                return x
            else:
                sr = self.transformer(x_mu.permute(1, 0, 2), acquirer=acquirer, lang=lang, zi_bool='mu',inputlayers = 1, text_visual=text_visual).permute(1, 0, 2)
                sa = self.transformer(x_sigma.permute(1, 0, 2), acquirer=acquirer, lang=lang, zi_bool='sigma',inputlayers = 1,text_visual = text_visual).permute(1, 0, 2)

                sr = self.ln_mu(sr).type(self.dtype)
                sa = self.ln_sigma(sa).type(self.dtype)

                sr = sr[torch.arange(sr.shape[0]), (text==102).nonzero()[:,1]] @ self.mu_projection

                sa = F.avg_pool1d(sa.permute(0, 2, 1), sa.size(1)).squeeze(2)

                if istrain:
                    adv_loss = 0
                    real_idx = 1
                    fake_idx = 0
                    # update discriminator
                    match_input = torch.cat((sa, torch.flip(src_feats, dims=[0])), -1)
                    unmatch_input = torch.cat((sa,src_feats), -1)                             #torch.flip(logvar, dims=[0]))
                    real_loss, fake_loss, real_acc, fake_acc = self.AdvAgent.update(match_input.detach(),
                                                                unmatch_input.detach(), real_idx, fake_idx)
                    # print(f"real_acc:{real_acc}, fake_acc:{fake_acc}, sum:{real_acc+fake_acc}")
                    adv_loss = self.AdvAgent.gen_loss(match_input, unmatch_input, real_idx, fake_idx)
                    return sr, sa, adv_loss , src_feats
                else:
                    return sr, sa
        else:
            x = x.permute(1, 0, 2)  # NLD -> LND
            xes = self.transformer(x, return_all_layers=True, acquirer=acquirer)
            # xes = self.transformer(x, return_all_layers=True)
            xes = [xes[layer] for layer in layers]
            xes = [x.permute(1, 0, 2) for x in xes] # NLD -> LND
            if acquirer and tokenize:
                xes = [x[torch.arange(x.shape[0]),  (text==102).nonzero()[:,1]] if i != len(layers)-1 else self.project_hidden(x, text, mbert=True) for i, x in enumerate(xes)]
            else:
                xes = [x[torch.arange(x.shape[0]), text.argmax(dim=-1)] if i != len(layers)-1 else self.project_hidden(x, text) for i, x in enumerate(xes)]
            return xes

    def project_hidden(self, x, text, mbert=False):
        x = self.ln_final(x).type(self.dtype)
        if mbert:
            x = x[torch.arange(x.shape[0]), (text==102).nonzero()[:,1]] @ self.text_projection
        else:

            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, acquirer=False, d_acquirer_hidden=256, m_acquirer=False, langs=None, skip=False, init_mbert_embedding=True,stage='CLA'):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        acquirer=acquirer, d_acquirer_hidden=d_acquirer_hidden,
        m_acquirer=m_acquirer, langs=langs, skip=skip, init_mbert_embedding=init_mbert_embedding,stage=stage
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict, strict=False)
    return model.eval()


