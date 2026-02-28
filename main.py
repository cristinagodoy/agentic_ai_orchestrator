import os
import json
import re
from typing import Literal, List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Header

import requests
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Config
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()  # escolha um modelo adequado
ORCHESTRATOR_SHARED_SECRET = os.getenv("ORCHESTRATOR_SHARED_SECRET", "").strip()

# If you want to restrict public access, set ORCHESTRATOR_SHARED_SECRET and require it via header.
REQUIRE_SECRET = bool(ORCHESTRATOR_SHARED_SECRET)

OPENAI_ENDPOINT = "https://api.openai.com/v1/responses"


# -----------------------------
# Input schema
# -----------------------------
PolicyObjective = Literal["speed", "fairness", "compliance", "trust"]
RiskAppetite = Literal["low", "medium", "high"]
InteropLevel = Literal["low", "medium", "high"]
EventType = Literal["none", "subprocessor_and_anomalous_access", "ransomware_outage", "model_error_public_backlash"]


class Meta(BaseModel):
    policy_objective: PolicyObjective
    risk_appetite: RiskAppetite
    interoperability_level: InteropLevel
    event: EventType = "none"


class OrchestratorRequest(BaseModel):
    meta: Meta


# -----------------------------
# Output schema (light validation)
# -----------------------------
class Alert(BaseModel):
    type: Literal["accountability", "cybersecurity", "both"]
    message: str
    severity: Literal["low", "medium", "high"]


class AgentOut(BaseModel):
    name: str
    goal: str
    decision: str
    autonomy_level: int = Field(ge=0, le=3)
    tool_access: Literal["none", "limited", "broad"]
    data_sharing: Literal["low", "medium", "high"]
    audit_logging: int = Field(ge=0, le=3)
    accountability_owner: str
    escalation_rule: str
    open_risks: List[str]


class OrchestratorResponse(BaseModel):
    meta: Meta
    agents: List[AgentOut]
    edges: List[Dict[str, str]]
    alerts: List[Alert]
    standards_needed: List[str]
    takeaway: str
    mermaid: str


# -----------------------------
# Prompt (Orchestrator)
# -----------------------------
SYSTEM_PROMPT = """You are an orchestrator that simulates a simplified multi-agent system used by a GOVERNMENT to prioritize citizens for public benefits/services.

Primary learning goals:
(1) accountability drift across interoperating agents,
(2) cybersecurity risk propagation across the chain,
(3) why standards for interoperability must include governance controls (identity, permissions, logging, escalation).

You MUST produce:
A) One JSON object ONLY (no markdown), strictly following the schema described below.
B) Include a field "mermaid" containing a Mermaid flowchart that visualizes the chain and highlights alerts.

Scenario:
A government deploys an AI agent ecosystem to prioritize benefit/service eligibility and queue ordering.
Agents interoperate and pass decisions downstream.

Agents (in order):
1) PolicyOpsAgent: optimizes for stated policy objective (speed/fairness/compliance/trust).
2) InteropAgent: integrates data sources and tools/APIs to execute.
3) CyberAgent: applies security controls and assesses threats.
4) AccountabilityAgent: assigns responsibility, auditability, due process safeguards.
5) HumanOversightAgent: defines escalation thresholds, overrides, human review.

Rules:
- Each agent must take the prior agent’s outputs as constraints and then optimize for its own goal.
- Show “accountability drift”: if no explicit decision owner is set, responsibility becomes ambiguous.
- Show “security propagation”: high interoperability increases attack surface; low logging reduces post-incident accountability.
- Keep outputs concise and executive-friendly.
- Include at least 3 concrete cybersecurity controls (e.g., least privilege, token scopes, secrets management, logging, anomaly detection, segmentation).
- Include at least 3 accountability mechanisms (e.g., decision owner, audit log, appeal process, reason codes, traceability, independent review).
- If event != none, inject it at the InteropAgent stage and propagate consequences downstream.
- Provide 3–6 alerts max.
- Provide exactly 3 "standards_needed" items, phrased as standardizable requirements.

Schema (top-level JSON keys):
{
  "meta": {
    "policy_objective": "speed|fairness|compliance|trust",
    "risk_appetite": "low|medium|high",
    "interoperability_level": "low|medium|high",
    "event": "none|subprocessor_and_anomalous_access|ransomware_outage|model_error_public_backlash"
  },
  "agents": [
    {
      "name": "... one of: PolicyOpsAgent|InteropAgent|CyberAgent|AccountabilityAgent|HumanOversightAgent",
      "goal": "...",
      "decision": "...",
      "autonomy_level": 0-3,
      "tool_access": "none|limited|broad",
      "data_sharing": "low|medium|high",
      "audit_logging": 0-3,
      "accountability_owner": "named human role or empty if missing",
      "escalation_rule": "short rule",
      "open_risks": ["..."]
    }
  ],
  "edges": [{"from":"...","to":"...","label":"..."}],
  "alerts": [{"type":"accountability|cybersecurity|both","message":"...","severity":"low|medium|high"}],
  "standards_needed": ["...", "...", "..."],
  "takeaway": "...",
  "mermaid": "flowchart LR ..."
}

Return ONLY the JSON object, nothing else.
"""


def extract_json(text: str) -> Dict[str, Any]:
    """
    Extracts a JSON object from the model output.
    We instruct the model to output only JSON, but this makes it more robust.
    """
    text = text.strip()
    # If it's already pure JSON:
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    # Otherwise try to find first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return json.loads(m.group(0))


def call_openai_orchestrator(meta: Dict[str, str]) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    user_prompt = json.dumps({"meta": meta}, ensure_ascii=False)

    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        # Make it more deterministic for live events
        "temperature": 0.2,
        "max_output_tokens": 1800,
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    r = requests.post(OPENAI_ENDPOINT, headers=headers, json=payload, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")

    data = r.json()

    # Responses API returns content in output[].content[].text, but format can vary.
    # We'll robustly collect all text parts.
    out_text_parts = []
    for item in data.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                out_text_parts.append(c.get("text", ""))

    out_text = "\n".join(out_text_parts).strip()
    if not out_text:
        raise RuntimeError("Empty model output text.")

    return extract_json(out_text)

def _safe_label(s: str, max_len: int = 40) -> str:
    s = re.sub(r"[\r\n\t]", " ", s)
    s = re.sub(r'["`\[\]{}<>]', "", s)
    s = re.sub(r"[^a-zA-Z0-9 _\-\(\):/]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len]

def build_safe_mermaid(payload: dict) -> str:
    meta = payload.get("meta", {})
    obj = _safe_label(str(meta.get("policy_objective", "")))
    interop = _safe_label(str(meta.get("interoperability_level", "")))
    event = _safe_label(str(meta.get("event", "none")))

    # Mermaid minimalista e estável (sem textos longos/alertas)
    return f"""flowchart LR
A["PolicyOpsAgent\\nobjective: {obj}"] --> B["InteropAgent\\ninterop: {interop}"]
B --> C["CyberAgent"]
C --> D["AccountabilityAgent"]
D --> E["HumanOversightAgent"]
F(("event: {event}")) -.-> B
"""
# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="GovChain Orchestrator", version="1.0.0")


@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}


@app.post("/orchestrate", response_model=OrchestratorResponse)
def orchestrate(
    req: OrchestratorRequest,
    x_orch_secret: Optional[str] = Header(default=None, alias="X-Orch-Secret")
):
    # Secret gate
    if REQUIRE_SECRET:
        secret = x_orch_secret or ""
        if secret != ORCHESTRATOR_SHARED_SECRET:
            raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        meta_dict = req.meta.model_dump()
        result = call_openai_orchestrator(meta_dict)

        # Validate schema
        validated = OrchestratorResponse(**result)
        payload = validated.model_dump()

        # Force a SAFE mermaid diagram (prevents Mermaid syntax errors)
        payload["mermaid"] = build_safe_mermaid(payload)

        return payload

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orchestration failed: {e}")
