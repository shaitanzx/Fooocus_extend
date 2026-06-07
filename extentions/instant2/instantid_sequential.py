import os
import math
import torch
import numpy as np
import cv2
import PIL.Image

# 🔧 ИМПОРТ ТВОЕГО To_KV
from modules.patch import To_KV

# ─────────────────────────────────────────────────────────────────────────────
# Custom Attention (как в оригинальном InstantID)
# ─────────────────────────────────────────────────────────────────────────────
class CustomAttention(torch.nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        
        self.to_q = torch.nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = torch.nn.Linear(dim, inner_dim * 2, bias=False)  # K и V вместе
        self.to_out = torch.nn.Linear(inner_dim, dim)

    def forward(self, x, context=None):
        # x: [B, N, D], context: [B, M, D]
        if context is None:
            context = x
        
        x = self.norm1(x)
        context = self.norm2(context)
        
        B, N, D = x.shape
        M = context.shape[1]
        
        # Проекции
        q = self.to_q(x)  # [B, N, inner_dim]
        kv = self.to_kv(context)  # [B, M, inner_dim*2]
        k, v = torch.chunk(kv, 2, dim=-1)  # [B, M, inner_dim] each
        
        # Reshape для multi-head
        h = self.heads
        q = q.reshape(B, N, h, -1).transpose(1, 2)  # [B, h, N, head_dim]
        k = k.reshape(B, M, h, -1).transpose(1, 2)
        v = v.reshape(B, M, h, -1).transpose(1, 2)
        
        # Attention
        sim = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        
        # Merge heads
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.to_out(out)

# ─────────────────────────────────────────────────────────────────────────────
# FeedForward (как в оригинале)
# ─────────────────────────────────────────────────────────────────────────────
class FeedForward(torch.nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, dim * mult, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(dim * mult, dim, bias=False),
        )
    def forward(self, x):
        return self.net(x)

# ─────────────────────────────────────────────────────────────────────────────
# Resampler (ТОЧНО как в оригинальном InstantID)
# ─────────────────────────────────────────────────────────────────────────────
class Resampler(torch.nn.Module):
    def __init__(self, dim=1280, depth=4, dim_head=64, heads=20, num_queries=16,
                 embedding_dim=512, output_dim=2048, ff_mult=4):
        super().__init__()
        self.num_queries = num_queries
        self.latents = torch.nn.Parameter(torch.randn(1, num_queries, dim))
        self.proj_in = torch.nn.Linear(embedding_dim, dim)
        self.proj_out = torch.nn.Linear(dim, output_dim)
        self.norm_out = torch.nn.LayerNorm(output_dim)
        
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(torch.nn.ModuleList([
                CustomAttention(dim, heads, dim_head),
                FeedForward(dim, ff_mult)
            ]))

    def forward(self, x):
        # x: [B, M, embedding_dim]
        h = self.latents.repeat(x.shape[0], 1, 1)  # [B, num_queries, dim]
        x = self.proj_in(x)  # [B, M, dim]
        
        for attn, ff in self.layers:
            # Cross-attention: h queries, x context
            h = attn(h, context=x) + h
            h = ff(h) + h
        
        return self.norm_out(self.proj_out(h))  # [B, num_queries, output_dim]

# ─────────────────────────────────────────────────────────────────────────────
# InstantIDAdapter
# ─────────────────────────────────────────────────────────────────────────────
class InstantIDAdapter(torch.nn.Module):
    def __init__(self, state_dict, cross_attention_dim=1280, output_cross_attention_dim=2048,
                 clip_embeddings_dim=512, clip_extra_context_tokens=16):
        super().__init__()
        self.image_proj_model = Resampler(
            dim=cross_attention_dim,           # 1280 (внутренний)
            depth=4,
            dim_head=64,
            heads=20,                           # 1280 // 64
            num_queries=clip_extra_context_tokens,  # 16
            embedding_dim=clip_embeddings_dim,      # 512
            output_dim=output_cross_attention_dim,  # динамический
            ff_mult=4
        )
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=False)
        
        self.ip_layers = To_KV(cross_attention_dim)
        self.ip_layers.load_state_dict_ordered(state_dict["ip_adapter"])

    @torch.inference_mode()
    def get_image_embeds(self, clip_embed, clip_embed_zeroed):
        return self.image_proj_model(clip_embed), self.image_proj_model(clip_embed_zeroed)

# ─────────────────────────────────────────────────────────────────────────────
# Keypoints Drawing
# ─────────────────────────────────────────────────────────────────────────────
def _draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)
    h, w, _ = image_pil.shape
    out_img = np.zeros([h, w, 3])
    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]
        x, y = kps[index][:, 0], kps[index][:, 1]
        length = ((x[0]-x[1])**2 + (y[0]-y[1])**2)**0.5
        angle = math.degrees(math.atan2(y[0]-y[1], x[0]-x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))),
                                   (int(length/2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)
    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        out_img = cv2.circle(out_img.copy(), (int(kp[0]), int(kp[1])), 10, color, -1)
    return PIL.Image.fromarray(out_img.astype(np.uint8))

# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_instantid_adapter(adapter_path: str, cross_attention_dim: int = 1280):
    print(f"[InstantID] 📦 Loading adapter: {adapter_path}")
    
    if adapter_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        raw = load_file(adapter_path, device="cpu")
    else:
        raw = torch.load(adapter_path, map_location="cpu")

    clean = {"image_proj": {}, "ip_adapter": {}}

    def clean_key(k):
        for prefix in ["diffusion_model.", "model.", "net.", "module.", "image_proj.", "ip_adapter."]:
            if k.startswith(prefix):
                k = k[len(prefix):]
        return k

    # Распаковка вложенных структур
    if len(raw) == 1 and isinstance(list(raw.values())[0], dict):
        raw = list(raw.values())[0]

    for k, v in raw.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                ck = clean_key(sub_k)
                if any(x in sub_k for x in ['image_proj', 'latents', 'proj_in', 'proj_out', 'norm_out', 'layers', 'to_q', 'to_kv', 'to_out', 'norm1', 'norm2']):
                    clean['image_proj'][ck] = sub_v
                elif any(x in sub_k for x in ['ip_adapter', 'to_k_ip', 'to_v_ip']):
                    clean['ip_adapter'][ck] = sub_v
        else:
            ck = clean_key(k)
            if any(x in k for x in ['image_proj', 'latents', 'proj_in', 'proj_out', 'norm_out', 'layers', 'to_q', 'to_kv', 'to_out', 'norm1', 'norm2']):
                clean['image_proj'][ck] = v
            elif any(x in k for x in ['ip_adapter', 'to_k_ip', 'to_v_ip']):
                clean['ip_adapter'][ck] = v

    if not clean["ip_adapter"]:
        print(f"[InstantID] ❌ Не найдены веса ip_adapter!")
        print(f"[InstantID] 🔍 Первые ключи: {list(raw.keys())[:10]}")
        raise KeyError("Adapter keys mismatch")

    # 🔧 Определяем output_dim как shape[1] (вход в Linear = выход Resampler)
    target_key = next((key for key in clean["ip_adapter"] if "to_k_ip" in key), None)
    if target_key is None:
        print(f"[InstantID] 🔍 Ключи ip_adapter: {list(clean['ip_adapter'].keys())[:10]}")
        raise KeyError("to_k_ip not found")

    weight_tensor = clean["ip_adapter"][target_key]
    output_cross_attention_dim = weight_tensor.shape[1]  # ← shape[1] = in_features
    
    print(f"[InstantID] ✅ Adapter loaded. Internal dim: {cross_attention_dim}, Output dim: {output_cross_attention_dim}")
    print(f"[InstantID] 🔍 Weight shape: {weight_tensor.shape} [out={weight_tensor.shape[0]}, in={weight_tensor.shape[1]}]")

    adapter = InstantIDAdapter(
        clean,
        cross_attention_dim=cross_attention_dim,
        output_cross_attention_dim=output_cross_attention_dim,
        clip_embeddings_dim=512,
        clip_extra_context_tokens=16
    )
    return adapter

def init_face_analyzer(provider: str = "CUDA", det_size: int = 640, root_path: str = None):
    print(f"[InstantID] 👁️ Initializing FaceAnalysis (provider={provider})...")
    from insightface.app import FaceAnalysis
    if root_path is None:
        root_path = os.path.join(os.path.dirname(__file__), "models")
    analyzer = FaceAnalysis(name="antelopev2", root=root_path, providers=[f'{provider}ExecutionProvider'])
    analyzer.prepare(ctx_id=0, det_size=(det_size, det_size))
    print("[InstantID] ✅ FaceAnalyzer ready.")
    return analyzer

# ─────────────────────────────────────────────────────────────────────────────
# SEQUENTIAL STEPS
# ─────────────────────────────────────────────────────────────────────────────

def step3_extract_embedding(face_analyzer, image_pil) -> torch.Tensor:
    print("[InstantID] STEP 3: Extracting face embedding...")
    faces = face_analyzer.get(np.array(image_pil))
    if not faces:
        print("[InstantID] ⚠️ STEP 3: No face detected!")
        return None
    face = sorted(faces, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
    print(f"[InstantID] STEP 3: Face found. Score: {face['det_score']:.2f}")
    return torch.from_numpy(face['embedding']).unsqueeze(0)

def step4_extract_kps(face_analyzer, image_pil) -> PIL.Image:
    print("[InstantID] STEP 4: Extracting face keypoints...")
    faces = face_analyzer.get(np.array(image_pil))
    if not faces:
        print("[InstantID] ⚠️ STEP 4: No face for keypoints!")
        return None
    face = sorted(faces, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
    return _draw_kps(np.array(image_pil), face['kps'])

def step5_compute_embeds(adapter: InstantIDAdapter, face_embed: torch.Tensor, 
                         noise: float = 0.35) -> tuple[torch.Tensor, torch.Tensor]:
    print("[InstantID] STEP 5: Computing cond/uncond embeddings...")
    from ldm_patched.modules import model_management as mm
    device = mm.get_torch_device()
    dtype = torch.float16 if mm.should_use_fp16() else torch.float32
    
    adapter.to(device, dtype=dtype)
    face_embed = face_embed.to(device, dtype=dtype)
    
    if noise > 0:
        torch.manual_seed(int(torch.sum(face_embed).item()) % 1_000_000_007)
        clip_embed_zeroed = noise * torch.rand_like(face_embed)
        print(f"[InstantID] STEP 5: Applied noise {noise}")
    else:
        clip_embed_zeroed = torch.zeros_like(face_embed)
        print("[InstantID] STEP 5: Uncond = zeros")

    cond, uncond = adapter.get_image_embeds(face_embed, clip_embed_zeroed)
    print(f"[InstantID] STEP 5: Embeds ready. Shape: {cond.shape}")
    return cond, uncond

def step6_patch_unet(model, adapter: InstantIDAdapter, cond: torch.Tensor, 
                     uncond: torch.Tensor, weight: float = 0.8, 
                     start_at: float = 0.0, end_at: float = 1.0,
                     is_sdxl: bool = True) -> object:
    print("[InstantID] STEP 6: Patching UNet attention layers...")
    from ldm_patched.modules.attention import optimized_attention
    
    work_model = model.clone()
    to = work_model.model_options.setdefault("transformer_options", {})
    patches = to.setdefault("patches_replace", {}).setdefault("attn2", {})
    
    ms = model.model.model_sampling
    sigma_start = ms.percent_to_sigma(start_at)
    sigma_end = ms.percent_to_sigma(end_at)
    
    # Блоки для SDXL или SD 1.5
    if is_sdxl:
        blocks = [
            ("input", 2, 1), ("input", 5, 2), ("input", 8, 10),
            ("middle", 0, 10),
            ("output", 3, 1), ("output", 6, 2), ("output", 8, 10)
        ]
    else:
        # Оригинальные блоки из ComfyUI InstantID (SD 1.5)
        blocks = []
        for block_id in [4, 5, 7, 8]:
            indices = range(2) if block_id in [4, 5] else range(10)
            for idx in indices:
                blocks.append(("input", block_id, idx))
        for block_id in range(6):
            indices = range(2) if block_id in [3, 4, 5] else range(10)
            for idx in indices:
                blocks.append(("output", block_id, idx))
        for idx in range(10):
            blocks.append(("middle", 1, idx))

    number = 0
    for block_type, block_id, depth in blocks:
        for idx in range(depth):
            key = (block_type, block_id, idx)
            k_layer = adapter.ip_layers.to_kvs[number * 2]
            v_layer = adapter.ip_layers.to_kvs[number * 2 + 1]

            class InstantIDPatch:
                def __init__(self, k_w, v_w, c, u, w, ss, se):
                    self.k_w, self.v_w = k_w, v_w
                    self.c, self.u = c, u
                    self.w = w
                    self.s_start, self.s_end = ss, se
                def __call__(self, q, context, value, extra_options):
                    org = q.dtype
                    c_or_u = extra_options["cond_or_uncond"]
                    sigma = extra_options.get("sigmas", [999.])[0].item()
                    if sigma > self.s_start or sigma < self.s_end:
                        return q
                    out = optimized_attention(q, context, value, extra_options["n_heads"])
                    embeds = torch.cat([self.c if i == 0 else self.u for i in c_or_u], dim=0)
                    k_ip = torch.nn.functional.linear(embeds, self.k_w) * self.w
                    v_ip = torch.nn.functional.linear(embeds, self.v_w) * self.w
                    out_ip = optimized_attention(q, k_ip.to(org), v_ip.to(org), extra_options["n_heads"])
                    return (out + out_ip).to(dtype=org)
            
            patches[key] = InstantIDPatch(k_layer.weight, v_layer.weight, cond, uncond, weight, sigma_start, sigma_end)
            number += 1
            
    print(f"[InstantID] STEP 6: Registered {number} attention patches.")
    return work_model

def step7_apply_controlnet(controlnet, positive, negative, kps_image, cond, uncond, 
                           strength: float = 0.8, start_at: float = 0.0, end_at: float = 1.0) -> tuple:
    print("[InstantID] STEP 7: Applying ControlNet to conditioning...")
    kps_np = np.array(kps_image).astype(np.float32) / 255.0
    kps_tensor = torch.from_numpy(kps_np).permute(2, 0, 1).unsqueeze(0)
    
    def _patch_cond(conditioning, cross_embed):
        new_c = []
        for t in conditioning:
            d = t[1].copy()
            c_net = controlnet.copy().set_cond_hint(kps_tensor, strength, (start_at, end_at))
            d['control'] = c_net
            d['control_apply_to_uncond'] = False
            d['cross_attn_controlnet'] = cross_embed
            new_c.append([t[0], d])
        return new_c

    pos_out = _patch_cond(positive, cond)
    neg_out = _patch_cond(negative, uncond)
    print("[InstantID] STEP 7: Conditioning modified.")
    return pos_out, neg_out