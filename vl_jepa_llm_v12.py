# ============================================================
# vl_jepa_llm_v12.py  (AMD Evo-X2 / Ryzen CPU OPTIMIZED)
# ============================================================
# Goals for the AMD 128GB CPU "Training Dojo" + fast local inference:
# - CPU-first performance tuning (threads, MKLDNN, matmul precision)
# - No torchvision CPU transforms bottlenecks; use torch ops (F.interpolate)
# - Optional DINOv2 backbone via torch.hub (cached) OR local weights (if you patch)
# - Batch-friendly embedding + similarity (compare is multisample + batched)
# - Robust concept parsing (punctuation-safe)
# - Paths relative to this file (portable across Mac/PC)
# - Backward-compatible checkpoint loading (old full state_dict or new partial dict)
#
# Run (CPU):
#   OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 python3 vl_jepa_llm_v12.py
#
# Optional speed knobs:
#   VLJEPA_THREADS=24 VLJEPA_INTEROP=2 VLJEPA_MKLDNN=1 VLJEPA_FP32_MATMUL=high
#   ALLOW_INSECURE_SSL=1  (only if your cert chain is borked; avoid if possible)
# ============================================================

import os
import re
import json
import ssl
import math
import time
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F

# If you use LM Studio on the AMD box, keep this.
# Otherwise you can comment it out and run in "manual tools" mode.
try:
    from openai import OpenAI  # LM Studio/OpenAI-compatible
except Exception:
    OpenAI = None


# ----------------------------
# 0) Root + Files
# ----------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
BRAIN_FILE = os.path.join(ROOT, "brain_vector_v12.pth")
MEMORY_FILE = os.path.join(ROOT, "memory_vector_v12.json")

LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
LM_MODEL = os.getenv("LM_MODEL", "local-model")

IMG_RES = int(os.getenv("VLJEPA_IMG_RES", "128"))

# ----------------------------
# 1) CPU Optimization (Ryzen)
# ----------------------------
def _cpu_optimize():
    # Threading
    threads = int(os.getenv("VLJEPA_THREADS", os.getenv("OMP_NUM_THREADS", "24")))
    interop = int(os.getenv("VLJEPA_INTEROP", "2"))

    # Torch threading
    try:
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(interop)
    except Exception:
        pass

    # MKLDNN / oneDNN
    mkldnn_on = os.getenv("VLJEPA_MKLDNN", "1") == "1"
    try:
        torch.backends.mkldnn.enabled = mkldnn_on
    except Exception:
        pass

    # Matmul precision (CPU helps a bit; safe default)
    # values: "highest" | "high" | "medium" (depending on torch version)
    matmul_prec = os.getenv("VLJEPA_FP32_MATMUL", "high")
    try:
        torch.set_float32_matmul_precision(matmul_prec)
    except Exception:
        pass

    # Reduce overhead for inference
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        pass

_cpu_optimize()

# ----------------------------
# 2) Device (AMD box = CPU)
# ----------------------------
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"[OK] Device: {DEVICE}")
if DEVICE.type == "cpu":
    print(f"[OK] Torch threads: {torch.get_num_threads()} (interop {torch.get_num_interop_threads()})")


# ----------------------------
# 3) Vector World Definitions
# ----------------------------
SHAPE_PROPERTIES = {
    # basic
    "circle": {"sides": 0.0}, "oval": {"sides": 0.0},
    "triangle": {"sides": 3.0}, "square": {"sides": 4.0},
    "rectangle": {"sides": 4.0}, "diamond": {"sides": 4.0},
    "pentagon": {"sides": 5.0}, "hexagon": {"sides": 6.0},
    "octagon": {"sides": 8.0}, "star": {"sides": 10.0},
    "heart": {"sides": 2.0}, "crescent": {"sides": 2.0},
    "cross": {"sides": 12.0}, "sphere": {"sides": 1.0}, "cone": {"sides": 2.0},
    "cylinder": {"sides": 3.0}, "pyramid": {"sides": 5.0},
    "cube": {"sides": 6.0}, "cuboid": {"sides": 6.0},

    # composites (coarse)
    "person": {"sides": 0.0},
    "golem": {"sides": 0.0},
    "tree": {"sides": 0.0},
    "house": {"sides": 4.0},
    "sword": {"sides": 2.0},
    "quill": {"sides": 2.0},
}

COLORS = {
    "red": (255, 0, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0),
    "green": (0, 255, 0), "orange": (255, 165, 0), "purple": (128, 0, 128),
    "white": (255, 255, 255), "black": (50, 50, 50), "grey": (128, 128, 128),
    "pink": (255, 192, 203), "brown": (165, 42, 42), "cyan": (0, 255, 255),
    "magenta": (255, 0, 255), "lime": (50, 205, 50), "indigo": (75, 0, 130),
    "teal": (0, 128, 128), "violet": (238, 130, 238), "silver": (192, 192, 192),
    "gold": (255, 215, 0),
}

SIZES = {"small": 0.30, "medium": 0.55, "large": 0.85, "extra_large": 1.10}

POSITIONS = {
    "center": (64, 64),
    "left": (32, 64), "right": (96, 64),
    "top": (64, 32), "bottom": (64, 96),
    "top-left": (32, 32), "top-right": (96, 32),
    "bottom-left": (32, 96), "bottom-right": (96, 96),
}

C_LIST = list(COLORS.keys())
S_LIST = list(SHAPE_PROPERTIES.keys())
Z_LIST = list(SIZES.keys())

# ----------------------------
# 4) Visual Renderer (fast, deterministic)
# ----------------------------
def _radius_from_size(size_name: str) -> int:
    scale = SIZES.get(size_name, 0.55)
    return int((IMG_RES // 2) * scale * 0.45)

def draw_tensor(
    shape_name: str,
    color_name: str,
    size_name: str,
    loc: Tuple[int, int] = (64, 64),
    canvas: Optional[Image.Image] = None,
) -> torch.Tensor | Image.Image:
    """
    If canvas is None -> returns torch tensor [3,H,W] float in [0,1]
    If canvas is provided -> draws onto it and returns PIL Image
    """
    if canvas is None:
        img = Image.new("RGB", (IMG_RES, IMG_RES), "black")
    else:
        img = canvas

    draw = ImageDraw.Draw(img)
    rgb = COLORS.get(color_name, (100, 100, 100))
    cx, cy = loc
    r = _radius_from_size(size_name)
    name = (shape_name or "").lower()

    # composites
    if ("person" in name) or ("human" in name) or ("golem" in name):
        head_r = max(2, r // 3)
        draw.ellipse([cx - head_r, cy - r, cx + head_r, cy - r + (head_r * 2)], fill=rgb)
        body_w = max(2, r // 2)
        draw.rectangle([cx - body_w, cy - r + (head_r * 2), cx + body_w, cy + r // 2], fill=rgb)
        draw.rectangle([cx - body_w, cy + r // 2, cx - (body_w // 2), cy + r], fill=rgb)
        draw.rectangle([cx + (body_w // 2), cy + r // 2, cx + body_w, cy + r], fill=rgb)

    elif ("house" in name) or ("building" in name):
        draw.rectangle([cx - r, cy - r // 3, cx + r, cy + r], fill=rgb)
        draw.polygon([cx - r, cy - r // 3, cx + r, cy - r // 3, cx, cy - r], fill=rgb)

    elif ("tree" in name) or ("plant" in name) or ("bloom" in name):
        trunk_w = max(2, r // 3)
        draw.rectangle([cx - trunk_w, cy, cx + trunk_w, cy + r], fill=(139, 69, 19))
        draw.ellipse([cx - r, cy - r, cx + r, cy], fill=rgb)

    elif ("sword" in name) or ("quill" in name):
        w = max(1, r // 5)
        draw.rectangle([cx - w, cy - r, cx + w, cy + r], fill=rgb)
        h_w = max(2, r // 2)
        draw.rectangle([cx - h_w, cy + r // 2, cx + h_w, cy + r // 2 + w], fill=rgb)

    # basics
    elif ("circle" in name) or ("sphere" in name):
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgb)
    elif ("square" in name) or ("cube" in name) or ("brick" in name):
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=rgb)
    elif ("triangle" in name) or ("pyramid" in name):
        draw.polygon([cx, cy - r, cx - r, cy + r, cx + r, cy + r], fill=rgb)
    elif "star" in name:
        points = []
        for i in range(10):
            d = r if i % 2 == 0 else r / 2.5
            points.extend([
                cx + d * math.cos(i * math.pi / 5 - math.pi / 2),
                cy + d * math.sin(i * math.pi / 5 - math.pi / 2),
            ])
        draw.polygon(points, fill=rgb)
    else:
        draw.ellipse([cx - r, cy - int(r / 1.5), cx + r, cy + int(r / 1.5)], fill=rgb)

    if canvas is not None:
        return img

    arr = np.array(img).astype(np.float32) / 255.0
    return torch.tensor(arr).permute(2, 0, 1).contiguous()

def draw_scene(objects_list: List[Dict[str, Any]]) -> torch.Tensor:
    sorted_objs = sorted(objects_list, key=lambda x: x.get("layer", 0))
    canvas = Image.new("RGB", (IMG_RES, IMG_RES), "black")

    for obj in sorted_objs:
        pos_key = obj.get("position", "center")
        loc = POSITIONS.get(pos_key, (64, 64))
        canvas = draw_tensor(
            obj.get("shape", "sphere"),
            obj.get("color", "grey"),
            obj.get("size", "medium"),
            loc=loc,
            canvas=canvas,
        )

    arr = np.array(canvas).astype(np.float32) / 255.0
    return torch.tensor(arr).permute(2, 0, 1).contiguous()


# ----------------------------
# 5) Cortex (DINOv2 + adapter)
# ----------------------------
class PretrainedViT(nn.Module):
    """
    AMD CPU optimized:
    - Uses pure torch resize+normalize (stays in torch)
    - Frozen backbone
    """
    def __init__(self):
        super().__init__()

        # SSL opt-in only
        if os.getenv("ALLOW_INSECURE_SSL", "0") == "1":
            ssl._create_default_https_context = ssl._create_unverified_context

        model_name = os.getenv("VLJEPA_DINO", "dinov2_vits14")
        repo = os.getenv("VLJEPA_DINO_REPO", "facebookresearch/dinov2")

        print(f"   [INFO] Loading DINOv2 backbone: {repo}:{model_name} (torch.hub cache applies)")

        # torch.hub will cache under ~/.cache/torch/hub after first run
        self.backbone = torch.hub.load(repo, model_name, verbose=False)

        for p in self.backbone.parameters():
            p.requires_grad = False

        # register buffers for CPU normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,128,128] float in [0,1]
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        x = (x - self.mean) / self.std
        return self.backbone(x)  # [B,384] for vits14


class OmniJEPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = PretrainedViT()

        self.adapter = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )

        self.head_c = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, len(C_LIST)))
        self.head_s = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, len(S_LIST)))
        self.head_z = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, len(Z_LIST)))
        self.logic_sides = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor):
        raw = self.vision(x)
        feat = self.adapter(raw)
        return feat, self.head_c(feat), self.head_s(feat), self.head_z(feat), self.logic_sides(feat)


# ----------------------------
# 6) Agent Brain (fast batch embed + tools)
# ----------------------------
@dataclass
class Concept:
    color: str
    shape: str
    size: str = "medium"

class AgentBrain:
    def __init__(self):
        self.model = OmniJEPA().to(DEVICE)
        self.model.eval()

        self.memory: Dict[str, Dict[str, Any]] = {}
        self._embed_cache: Dict[str, torch.Tensor] = {}

        self.load_brain()

        # LLM client optional
        self.llm_client = None
        if OpenAI is not None and os.getenv("VLJEPA_USE_LLM", "1") == "1":
            try:
                self.llm_client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
            except Exception:
                self.llm_client = None

        self.system_prompt = """You are an AI with a Visual Cortex and Spatial Imagination.

TOOLS (JSON in a code block):

1) IMAGINE:
{ "action":"imagine", "objects":[
  {"shape":"tree","color":"green","size":"large","position":"left","layer":0},
  {"shape":"person","color":"grey","size":"medium","position":"right","layer":1}
]}

2) LEARN:
{ "action":"learn", "concept":"ruby", "definition":"red diamond", "size":"small" }

3) COMPARE:
{ "action":"compare", "concept_a":"ruby", "concept_b":"emerald" }

Return ONLY valid JSON in a ```json code block.
"""

        self.history = [{"role": "system", "content": self.system_prompt}]
        self.action_history: List[str] = []
        self.MAX_RETRIES = 3

    # --------- Loading/Saving ---------
    def _load_state_forgiving(self, state_dict: dict):
        model_sd = self.model.state_dict()
        filtered = {}
        for k, v in state_dict.items():
            if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape):
                filtered[k] = v
        self.model.load_state_dict(filtered, strict=False)

    def load_brain(self):
        # weights
        if os.path.exists(BRAIN_FILE):
            try:
                state = torch.load(BRAIN_FILE, map_location=DEVICE, weights_only=False)
                if isinstance(state, dict) and any(k in state for k in ["adapter", "head_c", "head_s", "head_z"]):
                    # partial dict format
                    if "adapter" in state:
                        self.model.adapter.load_state_dict(state["adapter"], strict=True)
                    if "head_c" in state:
                        self.model.head_c.load_state_dict(state["head_c"], strict=True)
                    if "head_s" in state:
                        try:
                            self.model.head_s.load_state_dict(state["head_s"], strict=True)
                        except Exception:
                            pass
                    if "head_z" in state:
                        self.model.head_z.load_state_dict(state["head_z"], strict=True)
                    if "logic_sides" in state:
                        self.model.logic_sides.load_state_dict(state["logic_sides"], strict=True)
                    print("[OK] Loaded cortex (partial dict).")
                else:
                    # old full state_dict
                    self._load_state_forgiving(state)
                    print("[OK] Loaded cortex (forgiving full state).")
            except Exception as e:
                print(f"[WARN] Failed to load cortex: {e}")

        # memory
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r") as f:
                    data = json.load(f)
                self.memory = data.get("aliases", {})
                print(f"[OK] Loaded memory: {len(self.memory)} concepts")
            except Exception as e:
                print(f"[WARN] Failed to load memory: {e}")

    def save_brain(self):
        # save memory
        with open(MEMORY_FILE, "w") as f:
            json.dump({"known": list(self.memory.keys()), "aliases": self.memory}, f)

        # save trainable parts only
        state = {
            "adapter": self.model.adapter.state_dict(),
            "head_c": self.model.head_c.state_dict(),
            "head_s": self.model.head_s.state_dict(),
            "head_z": self.model.head_z.state_dict(),
            "logic_sides": self.model.logic_sides.state_dict(),
        }
        torch.save(state, BRAIN_FILE)

    # --------- Loop detection ---------
    def detect_loop(self, new_action: str, new_sig: str) -> bool:
        sig = f"{new_action}:{new_sig}"
        self.action_history.append(sig)
        if len(self.action_history) > 10:
            self.action_history.pop(0)
        return self.action_history.count(sig) >= self.MAX_RETRIES

    # --------- Robust parsing ---------
    def _parse_definition(self, definition: str) -> Tuple[str, str]:
        tokens = re.findall(r"[a-zA-Z\-]+", (definition or "").lower())
        color = next((t for t in tokens if t in COLORS), "grey")

        shape = "sphere"
        for t in tokens:
            if t in SHAPE_PROPERTIES:
                shape = t
                break
        return color, shape

    # --------- Embedding (batch) ---------
    @torch.inference_mode()
    def embed_batch(self, imgs: torch.Tensor) -> torch.Tensor:
        imgs = imgs.to(DEVICE)
        feat, _, _, _, _ = self.model(imgs)
        return feat

    @torch.inference_mode()
    def embed_concept(self, concept_name: str, jitter: int = 0) -> torch.Tensor:
        if jitter == 0 and concept_name in self._embed_cache:
            return self._embed_cache[concept_name]

        c = self.memory[concept_name]
        loc = (64, 64)
        if jitter != 0:
            loc = (64 + random.randint(-jitter, jitter), 64 + random.randint(-jitter, jitter))

        img = draw_tensor(c["shape"], c["color"], c.get("size", "medium"), loc=loc)
        feat = self.embed_batch(img.unsqueeze(0))[0].cpu()

        if jitter == 0:
            self._embed_cache[concept_name] = feat
        return feat

    # --------- Tools ---------
    def visual_query(self, action: str, cmd_data: Dict[str, Any]) -> str:
        action = (action or "").lower()

        if action == "imagine":
            objs = cmd_data.get("objects", [])
            tensor = draw_scene(objs)

            with torch.inference_mode():
                _, _, _, _, psides = self.model(tensor.unsqueeze(0).to(DEVICE))
            return f"[SCENE] GENERATED: objects={len(objs)} complexity={psides.item():.2f}"

        if action == "learn":
            concept = cmd_data.get("concept")
            definition = cmd_data.get("definition")
            if not concept or not definition:
                return "Error: learn requires concept + definition."

            color, shape = self._parse_definition(definition)
            size = cmd_data.get("size", "medium")

            self.memory[concept] = {"color": color, "shape": shape, "size": size}
            self._embed_cache.pop(concept, None)
            self.save_brain()
            return f"[OK] Learned {concept} = {color} {shape} ({size})."

        if action == "compare":
            a = cmd_data.get("concept_a")
            b = cmd_data.get("concept_b")
            if a not in self.memory or b not in self.memory:
                return "Unknown concept(s)."

            samples = int(cmd_data.get("samples", int(os.getenv("VLJEPA_COMPARE_SAMPLES", "8"))))
            jitter = int(cmd_data.get("jitter", int(os.getenv("VLJEPA_COMPARE_JITTER", "6"))))

            imgs = []
            for _ in range(samples):
                ca = self.memory[a]
                cb = self.memory[b]
                la = (64 + random.randint(-jitter, jitter), 64 + random.randint(-jitter, jitter))
                lb = (64 + random.randint(-jitter, jitter), 64 + random.randint(-jitter, jitter))
                imgs.append(draw_tensor(ca["shape"], ca["color"], ca.get("size", "medium"), loc=la))
                imgs.append(draw_tensor(cb["shape"], cb["color"], cb.get("size", "medium"), loc=lb))

            batch = torch.stack(imgs, dim=0)
            feats = self.embed_batch(batch).cpu()

            sims = []
            for i in range(samples):
                f1 = feats[2 * i + 0]
                f2 = feats[2 * i + 1]
                sims.append(F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item())

            sim = float(np.mean(sims))
            return f"[SIM] Similarity({samples}): {sim:.4f}"

        return "Unknown action."

    # --------- Chat loop (optional) ---------
    def chat(self, user_input: str) -> str:
        if self.llm_client is None:
            return "LLM disabled/unavailable. Use tool JSON manually or set VLJEPA_USE_LLM=1 with LM Studio."

        self.history.append({"role": "user", "content": user_input})

        try:
            completion = self.llm_client.chat.completions.create(
                model=LM_MODEL,
                messages=self.history,
                temperature=0.6
            )
            resp = completion.choices[0].message.content

            match = re.search(r"```json\s*({.*?})\s*```", resp, re.DOTALL)
            if not match:
                self.history.append({"role": "assistant", "content": resp})
                return resp

            cmd = json.loads(match.group(1))
            if self.detect_loop(cmd.get("action", "none"), json.dumps(cmd, sort_keys=True)):
                return "System Alert: Loop detected. Stopping."

            res = self.visual_query(cmd.get("action", ""), cmd)

            self.history.append({"role": "assistant", "content": resp})
            self.history.append({"role": "user", "content": f"System: {res}"})

            final = self.llm_client.chat.completions.create(
                model=LM_MODEL,
                messages=self.history,
                temperature=0.2
            ).choices[0].message.content
            return final

        except Exception as e:
            return f"Error: {e}"


# ----------------------------
# 7) Manual mode helpers (no LLM)
# ----------------------------
def _manual_tools_demo(agent: AgentBrain):
    print("\nManual tools mode examples (paste as JSON):")
    print('  {"action":"learn","concept":"ruby","definition":"red diamond","size":"small"}')
    print('  {"action":"learn","concept":"emerald","definition":"green diamond","size":"small"}')
    print('  {"action":"compare","concept_a":"ruby","concept_b":"emerald","samples":8,"jitter":6}')
    print('  {"action":"imagine","objects":[{"shape":"house","color":"red","size":"large","position":"left","layer":0},{"shape":"tree","color":"green","size":"large","position":"right","layer":1}]}')


# ----------------------------
# 8) Entry Point
# ----------------------------
if __name__ == "__main__":
    agent = AgentBrain()

    print("\n[OK] VL-JEPA v12 (AMD CPU optimized) ONLINE.")
    print("   - Use LLM chat (LM Studio) if configured, OR paste tool JSON manually.")
    if agent.llm_client is None:
        _manual_tools_demo(agent)

    while True:
        try:
            user = input("\nYou: ").strip()
            if user.lower() in ("quit", "exit"):
                break

            # If user pastes raw JSON, run tool directly (fast / no LLM needed)
            if user.startswith("{") and user.endswith("}"):
                try:
                    cmd = json.loads(user)
                    print(agent.visual_query(cmd.get("action", ""), cmd))
                except Exception as e:
                    print(f"Bad JSON: {e}")
                continue

            # Otherwise, use chat loop if available
            print(agent.chat(user))

        except KeyboardInterrupt:
            break
