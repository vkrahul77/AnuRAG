"""
AnuRAG: Configuration Module
Centralized configuration for all model settings.

Change the model here to affect the entire system!
"""

import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# LLM PROVIDER SELECTION  ("gemini" or "claude")
# =============================================================================
# Switch between providers with a single variable!
# Everything else (database, embeddings, search) stays the same.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

# =============================================================================
# GEMINI MODEL CONFIGURATION
# =============================================================================

# Main reasoning/chat model
# Options:
#   - "gemini-2.0-flash"        (fastest, cheapest, good quality)
#   - "gemini-2.0-flash-lite"   (even faster/cheaper, lower quality)
#   - "gemini-1.5-flash"        (older but stable)
#   - "gemini-1.5-pro"          (best quality, slower, more expensive)
#   - "gemini-2.0-pro"          (newest pro model)
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")

# Embedding model (for vector search)
# Options:
#   - "gemini-embedding-001"     (current, 3072 native / 768 with output_dimensionality)
#   - "text-embedding-004"      (DEPRECATED â€” removed from API Feb 2026)
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

# Vision model (for image analysis)
# Usually same as chat model since Gemini is multimodal
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.0-flash")

# Contextualization model (for offline chunk contextualization)
# Use a cheap/fast model â€” this is a one-time offline pass over all chunks.
# gemini-2.0-flash is recommended: cheap, fast, sufficient for one-line summaries.
GEMINI_CONTEXT_MODEL = os.getenv("GEMINI_CONTEXT_MODEL", "gemini-2.0-flash")


# =============================================================================
# CLAUDE (ANTHROPIC) MODEL CONFIGURATION
# =============================================================================

# Main reasoning/chat model for Claude
# Options:
#   - "claude-sonnet-4-20250514"  (RECOMMENDED: best cost/intelligence ratio)
#   - "claude-opus-4-20250514"    (maximum reasoning power, 5x more expensive)
#   - "claude-3-5-haiku-latest"   (fast & cheap, good for simple queries)
#
# Cost comparison (per 1M tokens):
#   claude-sonnet-4:  $3 input  / $15 output
#   claude-opus-4:    $15 input / $75 output
#   claude-3.5-haiku: $0.80 input / $4 output
#   gemini-2.0-flash: $0.10 input / $0.40 output
#   gemini-2.0-pro:   $1.25 input / $5.00 output
CLAUDE_CHAT_MODEL = os.getenv("CLAUDE_CHAT_MODEL", "claude-sonnet-4-20250514")


# =============================================================================
# LOCAL EMBEDDINGS CONFIGURATION
# =============================================================================

# Use local embeddings instead of Gemini API (FREE, no rate limits!)
USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"

# Local embedding model (if USE_LOCAL_EMBEDDINGS is True)
# Options:
#   - "all-MiniLM-L6-v2"        (fast, 384 dimensions)
#   - "all-mpnet-base-v2"       (better quality, 768 dimensions)
#   - "multi-qa-mpnet-base-dot-v1" (optimized for Q&A)
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")


# =============================================================================
# COST CONFIGURATION (per million tokens)
# =============================================================================

# Model pricing (per million tokens)
MODEL_COSTS = {
    # --- Gemini models ---
    "gemini-2.0-flash": {
        "input": 0.10,   # $0.10 per 1M input tokens
        "output": 0.40,  # $0.40 per 1M output tokens
    },
    "gemini-2.0-flash-lite": {
        "input": 0.075,
        "output": 0.30,
    },
    "gemini-1.5-flash": {
        "input": 0.075,
        "output": 0.30,
    },
    "gemini-1.5-pro": {
        "input": 1.25,   # $1.25 per 1M input tokens
        "output": 5.00,  # $5.00 per 1M output tokens
    },
    "gemini-2.0-pro": {
        "input": 1.25,
        "output": 5.00,
    },
    # --- Claude (Anthropic) models ---
    "claude-sonnet-4-20250514": {
        "input": 3.00,   # $3.00 per 1M input tokens
        "output": 15.00, # $15.00 per 1M output tokens
    },
    "claude-opus-4-20250514": {
        "input": 15.00,  # $15.00 per 1M input tokens
        "output": 75.00, # $75.00 per 1M output tokens
    },
    "claude-3-5-haiku-latest": {
        "input": 0.80,   # $0.80 per 1M input tokens
        "output": 4.00,  # $4.00 per 1M output tokens
    },
}

def get_active_chat_model() -> str:
    """Get the active chat model name based on LLM_PROVIDER setting."""
    if LLM_PROVIDER.lower() in ("claude", "anthropic"):
        return CLAUDE_CHAT_MODEL
    return GEMINI_CHAT_MODEL


def get_model_costs(model_name: str = None) -> dict:
    """Get cost configuration for a model."""
    if model_name is None:
        model_name = get_active_chat_model()
    # Try exact match, then prefix match for Claude versioned models
    if model_name in MODEL_COSTS:
        return MODEL_COSTS[model_name]
    for key in MODEL_COSTS:
        if model_name.startswith(key.rsplit('-', 1)[0]):
            return MODEL_COSTS[key]
    return MODEL_COSTS["gemini-2.0-flash"]


# =============================================================================
# RATE LIMITING CONFIGURATION
# =============================================================================

# Requests per minute (RPM) limits for free tier
RATE_LIMITS = {
    "gemini-2.0-flash": 15,
    "gemini-1.5-flash": 15,
    "gemini-1.5-pro": 2,
    "gemini-embedding-001": 1500,  # Embeddings have higher limits
}

# Delay between API calls (in seconds)
API_DELAY = float(os.getenv("API_DELAY", "4.0"))  # 4 seconds = 15 RPM


# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# Default path to PDF papers
DEFAULT_PAPER_PATH = os.getenv(
    "PAPER_PATH", 
    "./papers"  # Set PAPER_PATH env var or place PDFs here
)

# Database paths
DB_BASE_PATH = os.getenv("DB_PATH", "../finalAgent_db")


# =============================================================================
# LOOKUP TABLE (LUT) CONFIGURATION
# =============================================================================

# Path to pygmid .mat lookup tables (user must set these)
# The LUT files should be in the same directory or specify absolute paths
LUT_NMOS_PATH = os.getenv("LUT_NMOS_PATH", "sg13_lv_nmos.mat")
LUT_PMOS_PATH = os.getenv("LUT_PMOS_PATH", "sg13_lv_pmos.mat")


def get_lut_info(nmos_path: str = None, pmos_path: str = None) -> str:
    """
    Introspect pygmid .mat LUT files and return a formatted string
    describing all available parameters, sweep ranges, and valid
    lookup combinations. This string is injected into the Stage 2
    system prompt so the LLM knows exactly what it can query.
    
    Returns:
        A multi-line string with LUT documentation for the LLM.
    """
    nmos_path = nmos_path or LUT_NMOS_PATH
    pmos_path = pmos_path or LUT_PMOS_PATH
    
    info_lines = []
    
    for label, path in [("NMOS", nmos_path), ("PMOS", pmos_path)]:
        if not os.path.exists(path):
            info_lines.append(f"  {label}: File not found at '{path}'")
            continue
        
        try:
            import scipy.io
            data = scipy.io.loadmat(path)
            
            # Find the struct name (first non-dunder key)
            struct_name = [k for k in data.keys() if not k.startswith('__')][0]
            struct = data[struct_name]
            fields = struct.dtype.names
            
            # Extract sweep ranges
            L_vals = struct['L'][0, 0].flatten()
            VGS_vals = struct['VGS'][0, 0].flatten()
            VDS_vals = struct['VDS'][0, 0].flatten()
            VSB_vals = struct['VSB'][0, 0].flatten()
            
            # Separate raw params from sweep vars
            sweep_vars = {'L', 'VGS', 'VDS', 'VSB', 'W', 'NFING', 'INFO', 'CORNER', 'TEMP'}
            raw_params = sorted([f for f in fields if f not in sweep_vars])
            
            info_lines.append(f"  {label} LUT: '{os.path.basename(path)}' (struct: {struct_name})")
            info_lines.append(f"    Raw parameters: {', '.join(raw_params)}")
            info_lines.append(f"    L range (um):   {L_vals.min():.3f} to {L_vals.max():.3f}  ({len(L_vals)} points: {', '.join(f'{v:.2f}' for v in L_vals[:5])}{'...' if len(L_vals)>5 else ''})")
            info_lines.append(f"    VGS range (V):  {VGS_vals.min():.2f} to {VGS_vals.max():.2f}")
            info_lines.append(f"    VDS range (V):  {VDS_vals.min():.2f} to {VDS_vals.max():.2f}")
            info_lines.append(f"    VSB range (V):  {VSB_vals.min():.2f} to {VSB_vals.max():.2f}")
        except Exception as e:
            info_lines.append(f"  {label}: Error reading '{path}': {e}")
    
    # Now add the valid pygmid lookup combinations
    info_lines.append("")
    info_lines.append("  VALID pygmid lookup() calls (use ONLY these):")
    info_lines.append("    n.lookup('ID_W',   GM_ID=gm_id, L=L)   # Current density [A/m]")
    info_lines.append("    n.lookup('GM_GDS', GM_ID=gm_id, L=L)   # Intrinsic gain gm/gds [V/V]")
    info_lines.append("    n.lookup('GM_W',   GM_ID=gm_id, L=L)   # Transconductance density [S/m]")
    info_lines.append("    n.lookup('GDS_W',  GM_ID=gm_id, L=L)   # Output conductance density [S/m]")
    info_lines.append("    n.lookup('GDS_ID', GM_ID=gm_id, L=L)   # gds/ID [1/V]")
    info_lines.append("    n.lookup('CGS_W',  GM_ID=gm_id, L=L)   # Gate-source cap density [F/m]")
    info_lines.append("    n.lookup('CGD_W',  GM_ID=gm_id, L=L)   # Gate-drain cap density [F/m]")
    info_lines.append("    n.lookup('CDD_W',  GM_ID=gm_id, L=L)   # Drain cap density [F/m]")
    info_lines.append("    n.lookup('CSS_W',  GM_ID=gm_id, L=L)   # Source cap density [F/m]")
    info_lines.append("    n.lookup('CGG_W',  GM_ID=gm_id, L=L)   # Total gate cap density [F/m]")
    info_lines.append("    n.lookup('GMB_W',  GM_ID=gm_id, L=L)   # Body transconductance density [S/m]")
    info_lines.append("    n.lookup('STH_W',  GM_ID=gm_id, L=L)   # Thermal noise PSD density")
    info_lines.append("    n.lookup('SFL_W',  GM_ID=gm_id, L=L)   # Flicker noise PSD density")
    info_lines.append("")
    info_lines.append("  INVALID lookups (DO NOT USE â€” will error or return empty):")
    info_lines.append("    n.lookup('VDSAT', ...)  # NOT in LUT. Use VDSsat = 2/(gm/ID) instead")
    info_lines.append("    n.lookup('VT', ...)     # NOT available at all L values")
    info_lines.append("    n.lookup('FUG', ...)    # NOT available at all L values")
    info_lines.append("    n.lookup('CDB_W', ...)  # NOT in LUT. Use Cdb = Cdd - Cgd instead")
    info_lines.append("")
    info_lines.append("  IMPORTANT UNITS:")
    info_lines.append("    - L is in MICRONS (e.g., L=0.13, L=0.5, L=1.0)")
    info_lines.append("    - W from lookup is in meters â†’ multiply by 1e6 for display in Âµm")
    info_lines.append("    - VDSsat = 2*ID/gm = 2/(gm/ID)  [strong-inversion square-law approx]")
    
    return "\n".join(info_lines)


# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

# Maximum turns for ReAct agent
MAX_AGENT_TURNS = int(os.getenv("MAX_AGENT_TURNS", "10"))

# Temperature for generation (0 = deterministic, 1 = creative)
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

# Max output tokens
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))


# =============================================================================
# PRINT CONFIGURATION ON LOAD
# =============================================================================

def print_config():
    """Print current configuration."""
    print("\n" + "="*50)
    print("AnuRAG Configuration")
    print("="*50)
    print(f"LLM Provider:    {LLM_PROVIDER}")
    if LLM_PROVIDER.lower() in ("claude", "anthropic"):
        print(f"Chat Model:      {CLAUDE_CHAT_MODEL}")
    else:
        print(f"Chat Model:      {GEMINI_CHAT_MODEL}")
    print(f"Embedding Model: {GEMINI_EMBEDDING_MODEL} (always Gemini)")
    print(f"Vision Model:    {GEMINI_VISION_MODEL} (always Gemini)")
    print(f"Local Embeddings: {USE_LOCAL_EMBEDDINGS}")
    if USE_LOCAL_EMBEDDINGS:
        print(f"  Local Model:   {LOCAL_EMBEDDING_MODEL}")
    print(f"API Delay:       {API_DELAY}s")
    print("="*50 + "\n")


if __name__ == "__main__":
    print_config()
