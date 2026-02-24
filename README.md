# 🧠 Synapse: Memory Engine for LLMs

**Synapse** is a hybrid memory engine designed to extend LLM intelligence with structured, indexed, and versioned long-term memory. It replaces raw history injection with a semantic search system (BM25 + Recency) ensuring high relevance and low token costs.

Available as both a **Model Context Protocol (MCP)** server and a standalone **CLI Engine**.

---

## ✨ Features
- **Hybrid Architecture**: Works as a plug-and-play MCP tool for Claude/Cursor or a background script.
- **Smart Scoring**: Uses BM25 crossed with Time Decay (Recency) to find what truly matters *now*.
- **Auto-Fragmentation**: Automatically splits large memories into chunks to fit context windows.
- **Deterministic & Fast**: SQLite persistence with an In-Memory RAM Index for <500ms lookups.
- **Zero-Dependency Core**: Pure Python 3 core engine.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+

### 1. Initial Setup
```bash
# Clone the repository
git clone https://github.com/your-username/synapse.git
cd synapse

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Standalone CLI Usage (Manual Control)
```bash
# Initialize the database
python3 -m synapse init

# Add a memory
python3 -m synapse add --path "project/stack" --title "Database" --content "Using SQLite and RAM Index."

# Search
python3 -m synapse search --prompt "what database are we using?"
```

### 3. MCP Server (For AI Assistants)
Connect Synapse to **Claude Desktop**, **Cursor**, or any MCP-compatible agent to give it permanent memory.

**Claude Desktop Configuration (`claude_desktop_config.json`):**
```json
{
  "mcpServers": {
    "synapse_memory": {
      "command": "/absolute/path/to/venv/bin/python",
      "args": ["-m", "synapse.mcp_server"],
      "env": {
         "PYTHONPATH": "/absolute/path/to/synapse_directory"
      }
    }
  }
}
```

---

## 🛠 Available Tools (via MCP)
- `search_memory`: Semantic and temporal search of facts.
- `store_memory`: Saves new facts with automatic category-based organization.
- `update_memory`: Updates existing facts, maintaining version history.
- `forget_memory`: Privacy tool to wipe specific categories.

---

## 🧹 Maintenance (CLI)
Keep your memory engine healthy with background tasks:
```bash
# Run Garbage Collector (obsoslescence and cleanup)
python3 -m synapse gc

# Database and Index optimization
python3 -m synapse optimize
```

## 🧪 Testing
```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

---

## 📄 License
MIT. See `LICENSE` for details.
