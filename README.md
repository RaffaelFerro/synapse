# Synapse: Long-term Memory for LLMs

Synapse is a lightweight memory engine designed to give agents and LLMs a permanent "brain." It solves the "goldfish memory" problem by storing important facts, decisions, and preferences in an organized way, retrieving only what is relevant to the current conversation.

Instead of stuffing your entire chat history into a prompt, Synapse uses a smart search system to find exactly what matters. This keeps your context window clean and your token costs low.

---

## How it works

Synapse is designed to be flexible and works in two main ways:

* **Plug-and-play (MCP)**: Connect it to any AI tool that supports the Model Context Protocol (such as Claude Desktop, Cursor, Zed, and others). It works instantly as a set of tools the AI can use to remember things about your projects and preferences.
* **As a Core Engine**: If you are building your own AI application, you can import Synapse directly into your Python project. It handles memory management, semantic search, and versioning without the need for complex vector database setups.

## Main Features

* **Smart Retrieval**: It doesn't just look for keywords. It combines relevance (BM25) with recency, meaning it prioritizes what you decided today over information from months ago.
* **Automatic Organization**: It manages hierarchical categories and automatically fragments long pieces of information to fit within LLM context limits.
* **Fast and Reliable**: Built on SQLite with an in-memory index, it retrieves memories in less than 500ms.
* **Zero Dependencies**: Written in pure Python 3 to be lightweight, easy to audit, and simple to integrate.

---

## Getting Started

### Prerequisites
* Python 3.10+

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/your-username/synapse.git
cd synapse

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (only required for the MCP server)
pip install -r requirements.txt
```

### 2. Standalone CLI Usage
You can manage your memory manually via the terminal:
```bash
# Initialize the database
python3 -m synapse init

# Add a memory
python3 -m synapse add --path "project/stack" --title "Database" --content "Using SQLite and RAM Index."

# Search
python3 -m synapse search --prompt "what database are we using?"
```

### 3. MCP Server Configuration
To use Synapse with tools like Claude Desktop or Cursor, add it to your configuration file:

**Example (`claude_desktop_config.json`):**
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

## Maintenance
Keep your memory engine healthy with built-in maintenance tools:

```bash
# Run Garbage Collector (clean up obsolete data)
python3 -m synapse gc

# Database and Index optimization
python3 -m synapse optimize
```

## Testing
```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

## License
MIT. See `LICENSE` for details.

