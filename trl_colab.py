# =============================================================================
# CropOps Agent — TRL/SFT (DEFINITIVE FIX)
# =============================================================================
# THE REAL BUG IN PREVIOUS VERSIONS:
#   The crop destinations (zone_1 vs zone_2) are RANDOMIZED per episode.
#   Previous prompts didn't tell the LLM which zone to use.
#   The LLM memorized ONE path, but couldn't match random per-episode zones.
#   Result: 0 deliveries every time.
#
# THIS VERSION FIXES IT BY:
#   1. Including the per-episode zone assignments in the prompt
#   2. Generating training data with the same per-episode prompt
#   3. The LLM learns: "given THESE specific zone assignments, do THIS path"
#   4. At eval, prompt includes the actual zones, model can match the right path
#
# How to run in Colab (T4 GPU):
#   !pip install -q "trl>=0.12,<0.18" "transformers>=4.46,<4.50" "datasets" "accelerate" "bitsandbytes" "peft"
#   exec(open("trl_colab_DEFINITIVE.py").read())
# Runtime: ~15-20 min on T4.
# =============================================================================

import os, re, sys, random
from collections import deque
import numpy as np
import torch
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 1. ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("server", exist_ok=True)
with open("server/__init__.py", "w") as f:
    f.write("")

with open("server/cropops_env_environment.py", "w") as f:
    f.write('''
import random, numpy as np
ALL_ACTIONS=["up","down","left","right","pickup","dropoff"]
ROAD=0;VILLAGE=1;WAREHOUSE=2;BLOCKED=3

class CropOpsEnvironment:
    GRID_SIZE=10;MAX_STEPS=80;DISRUPTIONS_PER_EPISODE=2
    VILLAGE_POSITIONS={1:(1,1),2:(1,8),3:(8,1),4:(8,8)}
    WAREHOUSE_POSITIONS={"zone_1":(4,2),"zone_2":(4,7)}
    AGENT_START=(1,1)
    STATIC_BLOCKED=[(3,4),(3,5),(3,6),(6,4),(6,5),(6,6)]
    def __init__(self):
        self.grid=None;self.agent_pos=None;self.crops=[]
        self.carried_crop=None;self.delivered_crops=[]
        self.current_time=0;self.done=False
        self.disruptions_log=[];self.active_blocked=set()
        self.breakdown_steps=0
        self.disruption_schedule=[];self._build_grid()
    def _build_grid(self):
        self.grid=np.zeros((self.GRID_SIZE,self.GRID_SIZE),dtype=int)
        for r,c in self.STATIC_BLOCKED:self.grid[r][c]=BLOCKED
        for vid,(r,c) in self.VILLAGE_POSITIONS.items():self.grid[r][c]=VILLAGE
        for z,(r,c) in self.WAREHOUSE_POSITIONS.items():self.grid[r][c]=WAREHOUSE
    def reset(self):
        self._build_grid()
        self.agent_pos=list(self.AGENT_START);self.carried_crop=None
        self.delivered_crops=[];self.current_time=0;self.done=False
        self.disruptions_log=[];self.active_blocked=set();self.breakdown_steps=0
        crop_types=["tomato","wheat","corn"];random.shuffle(crop_types)
        zones=["zone_1","zone_2",random.choice(["zone_1","zone_2"])];random.shuffle(zones)
        self.crops=[]
        for i,(ctype,vid,zone) in enumerate(zip(crop_types,[1,2,3],zones)):
            sp=random.randint(35,50)
            self.crops.append({"id":i+1,"type":ctype,"village_id":vid,
                "position":list(self.VILLAGE_POSITIONS[vid]),"spoilage_remaining":sp,
                "intended_zone":zone,"picked_up":False,"delivered":False})
        self.disruption_schedule=sorted(random.sample(range(20,self.MAX_STEPS-5),self.DISRUPTIONS_PER_EPISODE))
        return None
    def step(self,action):
        if self.done:return None,0.0,True,{}
        self.current_time+=1;reward=0.0
        for c in self.crops:
            if not c["delivered"]:c["spoilage_remaining"]=max(0,c["spoilage_remaining"]-1)
        if action in("up","down","left","right"):
            if self.breakdown_steps>0:self.breakdown_steps-=1;reward=-0.5
            else:
                moved=self._try_move(action);reward=-0.05 if moved else -0.5
        elif action=="pickup":reward=self._try_pickup()
        elif action=="dropoff":reward=self._try_dropoff()
        else:reward=-0.5
        if all(c["delivered"] for c in self.crops) or self.current_time>=self.MAX_STEPS:self.done=True
        return None,reward,self.done,{}
    def _try_move(self,action):
        r,c=self.agent_pos
        d={"up":(-1,0),"down":(1,0),"left":(0,-1),"right":(0,1)}
        dr,dc=d[action];nr,nc=r+dr,c+dc
        if not(0<=nr<self.GRID_SIZE and 0<=nc<self.GRID_SIZE):return False
        if self.grid[nr][nc]==BLOCKED or (nr,nc) in self.active_blocked:return False
        self.agent_pos=[nr,nc];return True
    def _try_pickup(self):
        pos=tuple(self.agent_pos)
        if self.carried_crop:return -0.5
        for crop in self.crops:
            if not crop["picked_up"] and not crop["delivered"] and tuple(crop["position"])==pos:
                if crop["spoilage_remaining"]<=0:return -2.0
                crop["picked_up"]=True;self.carried_crop=crop;return 8.0
        return -0.5
    def _try_dropoff(self):
        pos=tuple(self.agent_pos)
        if not self.carried_crop:return -0.5
        for zone,wpos in self.WAREHOUSE_POSITIONS.items():
            if tuple(wpos)==pos:
                crop=self.carried_crop;crop["delivered"]=True
                self.delivered_crops.append(crop);self.carried_crop=None
                if zone==crop["intended_zone"]:
                    return 30.0 if crop["spoilage_remaining"]>10 else 20.0
                return -8.0
        return -0.5
''')

for mod in list(sys.modules.keys()):
    if 'cropops' in mod or 'server' in mod:
        del sys.modules[mod]

from server.cropops_env_environment import CropOpsEnvironment
print("✅ Environment loaded")


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXPERT POLICY + HELPERS
# ─────────────────────────────────────────────────────────────────────────────
GRID_SIZE = 10
STATIC_BLOCKED_SET = {(3,4),(3,5),(3,6),(6,4),(6,5),(6,6)}
ACTION_DELTAS = {"up":(-1,0), "down":(1,0), "left":(0,-1), "right":(0,1)}


def bfs_path(start, goal, blocked):
    if start == goal: return []
    queue = deque([(start, [])]); visited = {start}
    while queue:
        (r, c), path = queue.popleft()
        for action, (dr, dc) in ACTION_DELTAS.items():
            nr, nc = r+dr, c+dc
            if not (0<=nr<GRID_SIZE and 0<=nc<GRID_SIZE): continue
            if (nr,nc) in blocked or (nr,nc) in visited: continue
            new_path = path + [action]
            if (nr,nc) == goal: return new_path
            visited.add((nr,nc)); queue.append(((nr,nc), new_path))
    return None


def expert_policy_for_env(env, max_actions=70):
    actions_taken = []
    while not env.done and len(actions_taken) < max_actions:
        if env.carried_crop is not None:
            target = env.WAREHOUSE_POSITIONS[env.carried_crop["intended_zone"]]
            target_action = "dropoff"
        else:
            pending = [c for c in env.crops if not c["picked_up"] and not c["delivered"]]
            if not pending: break
            pending.sort(key=lambda c: c["spoilage_remaining"])
            target = tuple(pending[0]["position"])
            target_action = "pickup"
        start = tuple(env.agent_pos)
        blocked = STATIC_BLOCKED_SET | env.active_blocked
        path = bfs_path(start, target, blocked)
        if path is None:
            path = bfs_path(start, target, STATIC_BLOCKED_SET)
        if path is None: break
        for action in path[:5]:
            if env.done or len(actions_taken) >= max_actions: break
            env.step(action); actions_taken.append(action)
            pos = tuple(env.agent_pos)
            for crop in env.crops:
                if (not crop["picked_up"] and not crop["delivered"]
                        and tuple(crop["position"]) == pos and env.carried_crop is None):
                    env.step("pickup"); actions_taken.append("pickup")
        if tuple(env.agent_pos) == target and not env.done:
            env.step(target_action); actions_taken.append(target_action)
    return actions_taken, len(env.delivered_crops)


# ─────────────────────────────────────────────────────────────────────────────
# 3. THE KEY FIX — prompt INCLUDES zone assignments per episode
# ─────────────────────────────────────────────────────────────────────────────
def build_prompt_for_env(env):
    """Build a prompt that tells the LLM which zone each crop goes to.

    THIS IS THE FIX. Previous versions had a generic prompt — the LLM had no way
    to know which warehouse to deliver to. Now we include the assignment.
    """
    # List crops with their target zones for this specific episode
    crop_descriptions = []
    for crop in env.crops:
        pos = crop["position"]
        zone = crop["intended_zone"]
        zone_pos = env.WAREHOUSE_POSITIONS[zone]
        crop_descriptions.append(
            f"- {crop['type']} at ({pos[0]},{pos[1]}) -> deliver to {zone} at ({zone_pos[0]},{zone_pos[1]})"
        )
    crops_text = "\n".join(crop_descriptions)

    return (
        f"You are a delivery agent in a 10x10 grid. Start at (1,1).\n"
        f"Crops to deliver this episode:\n{crops_text}\n"
        f"Output a comma-separated action sequence to deliver all crops. "
        f"Actions: up, down, left, right, pickup, dropoff."
    )


def run_actions_in_env_with_seed(actions, seed):
    """Run actions starting from an env reset with given seed."""
    random.seed(seed); np.random.seed(seed)
    env = CropOpsEnvironment(); env.reset()
    total = 0.0
    for a in actions:
        if env.done: break
        _, r, done, _ = env.step(a)
        total += r
        if done: break
    return total, len(env.delivered_crops)


def get_env_state_for_seed(seed):
    """Build env with seed, return (env, prompt) — env is in initial state."""
    random.seed(seed); np.random.seed(seed)
    env = CropOpsEnvironment(); env.reset()
    prompt = build_prompt_for_env(env)
    return env, prompt


ACTION_RE = re.compile(r'(?:^|[\s,])(up|down|left|right|pickup|dropoff)(?=$|[\s,.])',
                       re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# 4. GENERATE TRAINING DATA — paired (prompt, actions) per episode
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating 300 paired (prompt, expert-actions) trajectories...")
training_pairs = []
attempts = 0
while len(training_pairs) < 300 and attempts < 1500:
    seed = 1000 + attempts
    random.seed(seed); np.random.seed(seed)
    env = CropOpsEnvironment(); env.reset()
    prompt = build_prompt_for_env(env)
    # Reset env (BFS uses fresh state)
    random.seed(seed); np.random.seed(seed)
    env = CropOpsEnvironment(); env.reset()
    actions, deliveries = expert_policy_for_env(env, max_actions=30)
    attempts += 1
    if deliveries >= 1 and 8 <= len(actions) <= 30:
        training_pairs.append({
            "seed": seed,
            "prompt": prompt,
            "actions": actions,
            "deliveries": deliveries,
        })

avg_d = np.mean([t["deliveries"] for t in training_pairs])
print(f"✅ Got {len(training_pairs)} pairs (avg expert: {avg_d:.2f}/3 deliveries)")
print(f"\nSample training pair:")
print(f"PROMPT:\n{training_pairs[0]['prompt']}")
print(f"\nACTIONS:\n{', '.join(training_pairs[0]['actions'])}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. LOAD QWEN-1.5B with 4-bit + LoRA
# ─────────────────────────────────────────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"\nLoading {MODEL_NAME} with 4-bit quantization...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if not torch.cuda.is_available():
    raise SystemExit("No GPU. Runtime → Change runtime type → T4 GPU.")

bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.bfloat16,
    bnb_4bit_use_double_quant = True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto",
)
print(f"✅ Loaded on {torch.cuda.get_device_name(0)}")

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"✅ LoRA: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")


def format_for_sft(pair):
    msgs = [
        {"role": "user", "content": pair["prompt"]},     # ← per-episode prompt!
        {"role": "assistant", "content": ", ".join(pair["actions"])},
    ]
    return {"text": tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False)}


sft_examples = [format_for_sft(p) for p in training_pairs]
dataset = Dataset.from_list(sft_examples)
print(f"\nTraining dataset: {len(dataset)} examples (each with unique prompt)")


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAIN
# ─────────────────────────────────────────────────────────────────────────────
config = SFTConfig(
    output_dir="./cropops_definitive",
    learning_rate=2e-4,
    per_device_train_batch_size=2,
    num_train_epochs=4,
    logging_steps=5,
    save_strategy="no",
    report_to=[],
    gradient_accumulation_steps=4,
    max_length=512,
    packing=False,
    dataset_text_field="text",
    optim="paged_adamw_32bit",
)

trainer = SFTTrainer(
    model=model, args=config, train_dataset=dataset, processing_class=tokenizer,
)

step_losses = []
_orig_log = trainer.log
def _track_log(logs, *args, **kwargs):
    if isinstance(logs, dict) and "loss" in logs:
        step_losses.append(float(logs["loss"]))
    return _orig_log(logs, *args, **kwargs)
trainer.log = _track_log

print("\n" + "=" * 60)
print("CropOps — TRL SFT (DEFINITIVE — per-episode prompts)")
print("=" * 60)

trainer.train()
print("\n✅ Training complete\n")


# ─────────────────────────────────────────────────────────────────────────────
# 7. EVALUATION — give LLM the SAME kind of prompt it trained on
# ─────────────────────────────────────────────────────────────────────────────
def generate_actions_llm(model, tokenizer, prompt):
    """Generate actions for a SPECIFIC episode's prompt."""
    msgs = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            min_new_tokens=25,
            do_sample=True,
            temperature=0.3,
            top_p=0.85,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:],
                            skip_special_tokens=True)


# Sanity check
print("=" * 60)
print("SANITY CHECK — model generations with episode-specific prompts:")
print("=" * 60)
for i in range(3):
    seed = 5000 + i
    env, prompt = get_env_state_for_seed(seed)
    raw = generate_actions_llm(model, tokenizer, prompt)
    n_acts = len(ACTION_RE.findall(raw))
    print(f"\n[{i+1}] Episode seed={seed}, prompt mentions zones:")
    # Show what zones this episode wants
    zones_wanted = [c["intended_zone"] for c in env.crops]
    print(f"    Zones to deliver: {zones_wanted}")
    print(f"    Model output ({n_acts} actions): {raw[:200]}")
print("=" * 60)


# Full evaluation
print("\nEvaluating on 20 episodes...\n")
random_d = []
llm_d = []
llm_actions_count = []
sample_outputs = []

for i in range(20):
    seed = 5000 + i

    # Random
    rand = [random.choice(["up","down","left","right","pickup","dropoff"]) for _ in range(30)]
    _, d = run_actions_in_env_with_seed(rand, seed)
    random_d.append(d)

    # LLM with episode-specific prompt
    env, prompt = get_env_state_for_seed(seed)
    raw = generate_actions_llm(model, tokenizer, prompt)
    actions = [a.lower() for a in ACTION_RE.findall(raw)][:50]
    _, d = run_actions_in_env_with_seed(actions, seed)
    llm_d.append(d)
    llm_actions_count.append(len(actions))

    if i < 3:
        sample_outputs.append((raw[:200], d, len(actions)))


print("=" * 60)
print("FINAL RESULTS — TRL SFT (DEFINITIVE)")
print("=" * 60)
print(f"Random              Deliveries: {np.mean(random_d):.2f}/3")
print(f"LLM (Qwen-1.5B)     Deliveries: {np.mean(llm_d):.2f}/3   "
      f"Avg actions: {np.mean(llm_actions_count):.1f}")
print(f"Improvement (LLM-Random): +{np.mean(llm_d)-np.mean(random_d):.2f}")
print("=" * 60)

print("\nSample LLM outputs:")
for i, (raw, d, n) in enumerate(sample_outputs):
    print(f"  [{i+1}] deliveries={d}/3 actions={n}: {raw[:150]}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. PLOT
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

if step_losses:
    axes[0].plot(step_losses, color='steelblue', alpha=0.6)
    if len(step_losses) >= 5:
        sma = np.convolve(step_losses, np.ones(5)/5, mode='valid')
        axes[0].plot(range(4, len(step_losses)), sma, color='navy', linewidth=2,
                     label='5-step avg')
        axes[0].legend()
axes[0].set_xlabel('Logging step'); axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss'); axes[0].grid(alpha=0.3)

labels = ['Random', 'LLM (Qwen-1.5B)']
delivs = [np.mean(random_d), np.mean(llm_d)]
ax2 = axes[1]
bars = ax2.bar(labels, delivs, color=['gray', 'green'])
ax2.bar_label(bars, fmt='%.2f', padding=3)
ax2.set_ylabel('Avg Deliveries / 3')
ax2.set_title('CropOps — TRL SFT Results')
ax2.set_ylim(0, 3)
ax2.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('reward_curve_trl.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ reward_curve_trl.png saved")
