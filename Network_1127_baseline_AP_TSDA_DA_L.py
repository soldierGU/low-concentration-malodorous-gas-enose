# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# 深度可分离空洞卷积
# ------------------------------
class SeparableConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride=1, padding=None, dilation=1, bias=False):
        super().__init__()
        if padding is None:
            padding = dilation * (k//2)
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size=k, stride=stride,
                            padding=padding, dilation=dilation, groups=in_ch, bias=bias)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)

# ------------------------------
# 抗混叠下采样
# ------------------------------
class BlurDown(nn.Module):
    def __init__(self, ch, mode='third', anti_alias='avg'):
        super().__init__()
        assert mode in ('half','third','none')
        self.mode = mode
        if mode == 'none':
            self.pool = nn.Identity()
            self.blur = None
        else:
            if anti_alias == 'blur':
                self.blur = nn.Conv1d(ch, ch, 3, 1, 1, groups=ch, bias=False)
                with torch.no_grad():
                    self.blur.weight.zero_()
                    self.blur.weight[:,0,0] = 0.25
                    self.blur.weight[:,0,1] = 0.5
                    self.blur.weight[:,0,2] = 0.25
            else:
                self.blur = None
            if mode == 'half':
                self.pool = nn.AvgPool1d(2,2) if anti_alias in ('avg','blur') else nn.MaxPool1d(2,2)
            else:
                self.pool = nn.MaxPool1d(3,3)  # ×1/3
    def forward(self, x):
        if self.blur is not None:
            x = self.blur(x)
        return self.pool(x)

# ------------------------------
# 边界平移
# ------------------------------
def _shift_right_zpad(x: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0: return x
    B, C, T = x.shape
    pad = x.new_zeros(B, C, k)
    return torch.cat([pad, x[..., :T - k]], dim=-1)

def _shift_left_zpad(x: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0: return x
    B, C, T = x.shape
    pad = x.new_zeros(B, C, k)
    return torch.cat([x[..., k:], pad], dim=-1)

# ------------------------------
# UnifiedHyperDA: 统一的 Depth×Time 二维聚合
# ------------------------------
class UnifiedHyperDA(nn.Module):
    """
    一个模块里统一 Depth-DA（跨层）+ TS-DA（时移）：
      - mode='joint2d': 二维联合（可分离外积 wd(t,k) 与 wt(t,s)）
      - 训练更稳：内置门控 gamma_d/gamma_t（Sigmoid），初期趋近于无额外路径
      - depth_k=0 时严格退化到纯 TS-DA，且动态强度与原始实现对齐（init_scale_time=1.0）
    """
    def __init__(self, dim, rate=2, mode='joint2d', boundary='zpad',
                 dynamic_depth=True, dynamic_time=True,
                 depth_k=0, depth_use_softmax=True, time_use_softmax=True,
                 per_channel_depth=False,
                 init_scale_time=1.0, init_scale_depth=1e-3,
                 depth_tau=2.0, time_tau=1.0):
        super().__init__()
        assert boundary in ('zpad', 'roll')
        assert mode in ('joint2d', 'depth_then_time', 'time_then_depth')
        self.boundary = boundary
        self.mode = mode
        self.dim = dim

        # sizes
        self.rate = int(rate)
        self.S = 2*self.rate + 1
        self.depth_k = int(depth_k)
        self.K = self.depth_k + 1 if self.depth_k > 0 else 1

        # gates & temps
        self.gamma_d = nn.Parameter(torch.tensor(0.0))
        self.gamma_t = nn.Parameter(torch.tensor(0.0))
        self.depth_tau = depth_tau
        self.time_tau  = time_tau

        # depth weights
        self.depth_use_softmax = depth_use_softmax
        self.per_channel_depth = per_channel_depth
        self.dynamic_depth = dynamic_depth

        if self.per_channel_depth:
            self.alpha_d = nn.Parameter(torch.zeros(dim, self.K))   # (C,K)
            with torch.no_grad(): self.alpha_d[:,0].fill_(1.0)
        else:
            self.alpha_d = nn.Parameter(torch.zeros(self.K))         # (K,)
            with torch.no_grad(): self.alpha_d[0] = 1.0

        if self.dynamic_depth:
            if self.per_channel_depth:
                self.conv_d = nn.Conv1d(dim, dim*self.K, kernel_size=1, groups=dim, bias=False)
            else:
                self.ln_d = nn.LayerNorm(dim)
                self.proj_d = nn.Linear(dim, self.K, bias=False)
            self.scale_d = nn.Parameter(torch.tensor(init_scale_depth))

        # time/shift weights
        self.time_use_softmax = time_use_softmax
        self.dynamic_time = dynamic_time

        self.alpha_t = nn.Parameter(torch.zeros(self.S))
        with torch.no_grad(): self.alpha_t[self.rate] = 2.0  # center bias

        if self.dynamic_time:
            self.ln_t = nn.LayerNorm(dim)
            self.proj_t = nn.Linear(dim, self.S, bias=False)
            self.scale_t = nn.Parameter(torch.tensor(init_scale_time))  # 对齐原始行为（=1.0）

    # ---- boundary shift ----
    def _shift(self, x, k):
        if k == 0: return x
        if self.boundary == 'roll':
            return torch.roll(x, shifts=k, dims=-1)
        return _shift_right_zpad(x,k) if k>0 else _shift_left_zpad(x,-k)

    # ---- depth weights (B,*,T,K) ----
    def _depth_weights(self, x):  # x:(B,C,T)
        B,C,T = x.shape
        if self.dynamic_depth:
            if self.per_channel_depth:
                logits = self.conv_d(x)                                 # (B,C*K,T)
                logits = logits.view(B, C, self.K, T).transpose(2,3)    # (B,C,T,K)
                alpha = self.alpha_d.view(1, C, 1, self.K)              # (1,C,1,K)
                logits = alpha + self.scale_d * logits
            else:
                q = self.ln_d(x.transpose(1,2))                         # (B,T,C)
                logits = self.alpha_d.view(1,1,self.K) + self.scale_d * self.proj_d(q)  # (B,T,K)
                logits = logits.unsqueeze(1)                             # (B,1,T,K)
        else:
            if self.per_channel_depth:
                logits = self.alpha_d.view(1,C,1,self.K)
            else:
                logits = self.alpha_d.view(1,1,1,self.K)

        if self.depth_use_softmax:
            wd = torch.softmax(logits / self.depth_tau, dim=-1)
        else:
            wd = torch.tanh(logits); wd = wd.abs()
            wd = wd / wd.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return wd  # (B,1(or C),T,K)

    # ---- time weights (B,1,T,S) ----
    def _time_weights(self, x):  # x:(B,C,T)
        B,C,T = x.shape
        if self.dynamic_time:
            q = self.ln_t(x.transpose(1,2))                       # (B,T,C)
            logits = self.alpha_t.view(1,1,self.S) + self.scale_t * self.proj_t(q)   # (B,T,S)
        else:
            logits = self.alpha_t.view(1,1,self.S).expand(B,T,self.S)
        if self.time_use_softmax:
            wt = torch.softmax(logits / self.time_tau, dim=-1)
        else:
            wt = torch.tanh(logits); wt = wt.abs()
            wt = wt / wt.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        wt = wt.unsqueeze(1)  # (B,1,T,S)
        return wt

    def forward(self, x, depth_feats=None):  # x:(B,C,T)
        B,C,T = x.shape

        # depth stack: [current, history...], len=K（不足补零，超出截断）
        if self.depth_k > 0 and depth_feats is not None and len(depth_feats) > 0:
            feats = [x]
            need = self.K - 1
            src = list(depth_feats)
            if len(src) >= need: feats += src[:need]
            else: feats += src + [x.new_zeros(B,C,T)] * (need - len(src))
        else:
            feats = [x]

        if self.mode == 'joint2d':
            wd = self._depth_weights(x)     # (B,1(or C),T,K)
            wt = self._time_weights(x)      # (B,1,T,S)
            g_d = torch.sigmoid(self.gamma_d)
            g_t = torch.sigmoid(self.gamma_t)

            shifted = []
            for k in range(self.K):
                xk = feats[k]
                neigh = []
                for s in range(-self.rate, self.rate+1):
                    neigh.append(self._shift(xk, s))
                neigh = torch.stack(neigh, dim=-1)     # (B,C,T,S)
                shifted.append(neigh)
            Z = torch.stack(shifted, dim=-2)           # (B,C,T,K,S)

            Wd = wd.unsqueeze(-1)                      # (B,1(or C),T,K,1)
            Wt = wt.unsqueeze(-2)                      # (B,1,T,1,S)
            W  = (g_d * Wd + (1-g_d)) * (g_t * Wt + (1-g_t))

            y = (Z * W).sum(dim=(-1,-2))               # (B,C,T)
            return y

        elif self.mode == 'depth_then_time':
            wd = self._depth_weights(x); g_d = torch.sigmoid(self.gamma_d)
            stacked = torch.stack(feats, dim=-1)       # (B,C,T,K)
            if wd.size(1) == 1: wd = wd.expand(B,1,T,self.K)
            y = (stacked * (g_d*wd + (1-g_d))).sum(dim=-1)  # (B,C,T)
            wt = self._time_weights(y); g_t = torch.sigmoid(self.gamma_t)
            neigh = [self._shift(y,s) for s in range(-self.rate,self.rate+1)]
            neigh = torch.stack(neigh, dim=-1)         # (B,C,T,S)
            y = (neigh * (g_t*wt + (1-g_t))).sum(dim=-1)
            return y

        else:  # 'time_then_depth'
            wt = self._time_weights(x); g_t = torch.sigmoid(self.gamma_t)
            ys = []
            for k in range(self.K):
                neigh = [self._shift(feats[k], s) for s in range(-self.rate,self.rate+1)]
                neigh = torch.stack(neigh, dim=-1)     # (B,C,T,S)
                ys.append((neigh * (g_t*wt + (1-g_t))).sum(dim=-1))
            y_stack = torch.stack(ys, dim=-1)          # (B,C,T,K)
            wd = self._depth_weights(x); g_d = torch.sigmoid(self.gamma_d)
            if wd.size(1) == 1: wd = wd.expand(B,1,T,self.K)
            y = (y_stack * (g_d*wd + (1-g_d))).sum(dim=-1)
            return y

# ------------------------------
# 多尺度 + 统一聚合 Block
# ------------------------------
class MScale_TSDA_Block(nn.Module):
    """
    多尺度分支 (k=1/3/5) -> concat -> 1x1 fuse -> PreNorm -> UnifiedHyperDA(可关) -> Downsample(可去AA)
    带残差：匹配通道&下采样
    """
    def __init__(self, in_ch, out_ch,
                 ks=(1,3,5), dilations=(1,2,3),
                 ts_rate=2, ts_mode='joint2d', ts_boundary='zpad',
                 down='half',
                 # Depth-DA config:
                 depth_k=0, depth_dynamic=False, depth_use_softmax=True,
                 depth_per_channel=False, depth_init_scale=1e-3,
                 depth_tau=2.0,
                 # Time/TS-DA config:
                 time_dynamic=True, time_use_softmax=True,
                 time_init_scale=1.0, time_tau=1.0,
                 # baseline switches:
                 enable_tsda=True,
                 use_antialias=True):
        super().__init__()
        assert len(ks) == len(dilations)
        self.enable_tsda = enable_tsda

        # ---- 多尺度卷积分支（无 ECA/SE 等注意力）----
        self.branches = nn.ModuleList()
        mid = out_ch // len(ks)
        last_extra = out_ch - mid*(len(ks)-1)

        for i, (k,d) in enumerate(zip(ks, dilations)):
            ch = last_extra if i==len(ks)-1 else mid
            if k == 1:
                self.branches.append(nn.Sequential(
                    nn.Conv1d(in_ch, ch, 1, bias=False),
                    nn.BatchNorm1d(ch),
                    nn.GELU()
                ))
            else:
                self.branches.append(SeparableConv1d(in_ch, ch, k=k, dilation=d, bias=False))

        self.fuse = nn.Conv1d(out_ch, out_ch, 1, bias=False)
        self.fuse_bn = nn.BatchNorm1d(out_ch)

        self.pre_ln = nn.GroupNorm(1, out_ch)

        if self.enable_tsda:
            self.tsda = UnifiedHyperDA(
                dim=out_ch,
                rate=ts_rate,
                mode=ts_mode,
                boundary=ts_boundary,
                # Depth:
                depth_k=depth_k,
                dynamic_depth=depth_dynamic,
                depth_use_softmax=depth_use_softmax,
                per_channel_depth=depth_per_channel,
                init_scale_depth=depth_init_scale,
                depth_tau=depth_tau,
                # Time:
                dynamic_time=time_dynamic,
                time_use_softmax=time_use_softmax,
                init_scale_time=time_init_scale,
                time_tau=time_tau
            )
        else:
            self.tsda = None

        aa_mode = 'blur' if use_antialias else 'none'
        self.down = BlurDown(out_ch, mode=down, anti_alias=aa_mode)

        # 残差路径
        self.proj = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.res_down = BlurDown(out_ch, mode=down, anti_alias=aa_mode) if down!='none' else nn.Identity()
        self.out_act = nn.GELU()

    def forward(self, x, depth_feats=None):
        res = self.proj(x)
        outs = [b(x) for b in self.branches]
        x = torch.cat(outs, dim=1)
        x = self.fuse_bn(self.fuse(x))
        x = self.pre_ln(x)
        if self.tsda is not None:
            x = self.tsda(x, depth_feats=depth_feats)     # 统一聚合
        x = self.down(x)
        res = self.res_down(res)
        return self.out_act(x + res)

# ------------------------------
# 主干网络（可调 Block 数量 L）
# ------------------------------
class MScale_TSDDA_Net_1127(nn.Module):
    """
    400×16 输入：
    stem(×2下采样) -> L 个多尺度 Block(每块×1/2) -> GAP -> FC
    - 不再使用 SE/ECA 注意力
    - L = len(widths) 可调
    - baseline 开关: 去掉 anti-aliasing pooling、TSDA、DA(depth_k=0)
    """
    def __init__(self, num_channels=16, num_classes=8,
                 stem_ch=64, widths=(64, 96, 128),
                 ks=(1,3,5), dilations=(1,2,3),
                 ts_rate=2, ts_mode='joint2d', ts_boundary='zpad',
                 # Depth-DA 策略（按 Block，一般最后几层开启）:
                 depth_k_tuple=(0, 0, 1),
                 depth_dynamic_tuple=(False, False, False),
                 per_channel_depth=False,
                 # baseline 控制:
                 use_antialias=True,
                 use_tsda=True,
                 baseline=False):
        super().__init__()

        # ---- 规范化列表 & Block 数 ----
        self.widths = list(widths)
        L = len(self.widths)
        assert L >= 1, "widths 至少包含一个通道数（对应一个 Block）"

        def _to_list(x, name, L):
            # 允许：单个值 / 与 L 等长的序列
            if isinstance(x, (int, bool)):
                return [x] * L
            if len(x) == L:
                return list(x)
            raise ValueError(f"{name} 长度需为 1 或与 widths 相同 (L={L})")

        depth_k_list = _to_list(depth_k_tuple, "depth_k_tuple", L)
        depth_dyn_list = _to_list(depth_dynamic_tuple, "depth_dynamic_tuple", L)

        # baseline: 统一关闭 AA + TSDA + DA
        if baseline:
            use_antialias = False
            use_tsda = False
            depth_k_list = [0] * L
            depth_dyn_list = [False] * L

        self.depth_k_list = depth_k_list
        self.depth_dyn_list = depth_dyn_list
        self.per_channel_depth = per_channel_depth
        self.use_antialias = use_antialias
        self.use_tsda = use_tsda
        self.num_blocks = L

        # Stem（×2 下采样; AA 可关）
        stem_aa_mode = 'blur' if use_antialias else 'none'
        self.stem = nn.Sequential(
            nn.Conv1d(num_channels, stem_ch, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(stem_ch),
            nn.GELU(),
            BlurDown(stem_ch, mode='half', anti_alias=stem_aa_mode)  # 400->200
        )

        # ---- L 个 Block + 对齐 Adapter（上一层 → 当前层通道）----
        self.blocks = nn.ModuleList()
        self.adapters = nn.ModuleList()

        prev_ch = stem_ch
        for i, out_ch in enumerate(self.widths):
            blk = MScale_TSDA_Block(
                prev_ch, out_ch,
                ks=ks, dilations=dilations,
                ts_rate=ts_rate, ts_mode=ts_mode, ts_boundary=ts_boundary,
                down='half',
                depth_k=self.depth_k_list[i],
                depth_dynamic=self.depth_dyn_list[i],
                depth_use_softmax=True,
                depth_per_channel=self.per_channel_depth,
                depth_init_scale=1e-3,
                depth_tau=2.0,
                time_dynamic=True,
                time_use_softmax=True,
                time_init_scale=1.0,
                time_tau=1.0,
                enable_tsda=self.use_tsda,
                use_antialias=self.use_antialias
            )
            self.blocks.append(blk)
            # depth 对齐适配器：上一层通道 -> 当前层 out_ch
            self.adapters.append(nn.Conv1d(prev_ch, out_ch, 1, bias=False))
            prev_ch = out_ch

        final_ch = self.widths[-1]

        # 末端不再使用 SE/ ECA 注意力头
        self.head_attn = nn.Identity()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(final_ch, num_classes)

    @staticmethod
    def _align_to(x_src, C_to, T_to, adapter):
        x = F.adaptive_avg_pool1d(x_src, T_to)
        x = adapter(x)  # 通道投影
        return x

    def forward(self, x):
        # Stem
        x_cur = self.stem(x)   # (B, stem_ch, T1)

        # 逐 Block 前向
        for i, blk in enumerate(self.blocks):
            depth_feats = None
            if self.depth_k_list[i] >= 1 and self.use_tsda:
                T = x_cur.size(-1)
                C_to = self.widths[i]
                depth_feats = [ self._align_to(x_cur, C_to=C_to, T_to=T, adapter=self.adapters[i]) ]
            x_cur = blk(x_cur, depth_feats=depth_feats)

        xh = self.head_attn(x_cur)
        xh = self.gap(xh).squeeze(-1)
        return self.fc(xh)


if __name__ == "__main__":
    B, Cin, T = 2, 16, 400
    x = torch.randn(B, Cin, T)

    # 标准配置：3 个 Block + TS-DA + 最后一层开启 Depth-DA
    model = MScale_TSDDA_Net_1127(
        num_channels=Cin, num_classes=8,
        stem_ch=64, widths=(64,96,128,160),
        ks=(1,3,5), dilations=(1,2,3),
        ts_rate=2, ts_mode='joint2d', ts_boundary='zpad',
        depth_k_tuple=(0,1,2,3), depth_dynamic_tuple=(False,True,True,True),
        per_channel_depth=False,
        use_antialias=True,
        use_tsda=True,
        baseline=False
    )
    with torch.no_grad():
        y = model(x)
    print("normal:", y.shape)

    # 基线模型：去掉 anti-aliasing pooling、TSDA、DA
    baseline_model = MScale_TSDDA_Net_1127(
        num_channels=Cin, num_classes=8,
        stem_ch=64, widths=(64,96,128),
        ks=(1,3,5), dilations=(1,2,3),
        ts_rate=2, ts_mode='joint2d', ts_boundary='zpad',
        depth_k_tuple=(0,0,0), depth_dynamic_tuple=(False,False,False),
        per_channel_depth=False,
        baseline=True
    )
    with torch.no_grad():
        y2 = baseline_model(x)
    print("baseline:", y2.shape)
