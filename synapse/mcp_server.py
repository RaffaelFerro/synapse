import asyncio
import json
import logging
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from synapse.engine import SynapseEngine, SynapseError
from synapse.db import DEFAULT_DB_PATH

# Configure logging to stderr so it doesn't interfere with MCP's stdout communication
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("synapse-mcp")

# Initialize FastMCP Server
mcp = FastMCP("SynapseMemory")

# Load the engine. In a production MCP environment, the DB path could be passed via env vars.
# For simplicity, we use the default local db.
try:
    engine = SynapseEngine(DEFAULT_DB_PATH, debug=False)
    logger.info(f"SynapseEngine initialized with DB: {DEFAULT_DB_PATH}")
except Exception as e:
    logger.error(f"Failed to initialize SynapseEngine: {e}")
    sys.exit(1)


@mcp.tool()
def search_memory(prompt: str) -> str:
    """
    Pesquisa na memória do usuário usando busca semântica e temporal (BM25 + Recency).
    Retorna os fatos mais relevantes para o prompt, respeitando limites de tokens.
    
    Args:
        prompt: A string de busca, pergunta ou contexto para buscar memórias relacionadas.
    """
    try:
        logger.info(f"Buscando memórias para o prompt: '{prompt}'")
        result = engine.search(prompt)
        
        # Filtra memórias missing e formata a resposta para ser facilmente legível pela LLM
        memories = [m for m in result.context.get("memories", []) if not m.get("missing")]
        
        if not memories:
            return "Nenhuma memória relevante encontrada para este prompt."
            
        formatted_output = "Memórias Retornadas:\n\n"
        for mem in memories:
            path = mem.get("path", "desconhecido")
            title = mem.get("title") or "Sem título"
            content = mem.get("content", "")
            date = mem.get("updated_at", "")
            
            formatted_output += f"--- [{path}] {title} (Atualizado em: {date}) ---\n"
            formatted_output += f"{content}\n\n"
            
        return formatted_output
    except SynapseError as e:
        return f"Erro do Synapse: {e}"
    except Exception as e:
        logger.error(f"Erro inesperado em search_memory: {e}")
        return f"Erro interno ao buscar memórias: {e}"


@mcp.tool()
def store_memory(category_path: str, content: str, title: str = "") -> str:
    """
    Armazena um novo fato ou decisão na memória de longo prazo.
    As memórias são organizadas por caminhos de categoria (ex: 'projeto/arquitetura').
    Se o conteúdo for longo, a engine fará o split automaticamente.
    
    Args:
        category_path: Caminho hierárquico da categoria (ex: 'pessoal/gostos', 'trabalho/projeto_x').
        content: O fato, decisão ou informação a ser lembrada.
        title: (Opcional) Título curto para a memória.
    """
    try:
        logger.info(f"Salvando memória em '{category_path}'")
        mem_id = engine.add_memory(path=category_path, title=title, content=content)
        return f"Memória salva com sucesso! ID: {mem_id}"
    except SynapseError as e:
        return f"Erro ao salvar memória: {e}"
    except Exception as e:
        logger.error(f"Erro inesperado em store_memory: {e}")
        return f"Erro interno ao salvar memória: {e}"


@mcp.tool()
def update_memory(memory_id: str, new_content: str) -> str:
    """
    Atualiza o conteúdo de uma memória existente, criando uma nova versão no histórico.
    Use isso para corrigir informações desatualizadas ou expandir fatos (conflict resolution).
    
    Args:
        memory_id: O ID exato da memória (retornado em store_memory ou search_memory).
        new_content: O novo conteúdo completo que substituirá o antigo.
    """
    try:
        logger.info(f"Atualizando memória '{memory_id}'")
        engine.update_memory(memory_id, content=new_content)
        return f"Memória {memory_id} atualizada com sucesso para uma nova versão."
    except SynapseError as e:
        return f"Erro ao atualizar memória: {e}"
    except Exception as e:
        logger.error(f"Erro inesperado em update_memory: {e}")
        return f"Erro interno ao atualizar memória: {e}"


@mcp.tool()
def forget_memory(category_prefix: str) -> str:
    """
    Ferramenta de privacidade (Wipe). Deleta todas as memórias que começam com um dado prefixo de categoria.
    As memórias correspondentes são marcadas como completamentes deletadas do banco.
    ATENÇÃO: Ação irreversível.
    
    Args:
        category_prefix: Caminho da categoria a ser varrida (ex: 'pessoal/segredos').
    """
    try:
        prefix = engine._normalize_path(category_prefix)
        if not prefix:
            return "Erro: prefixo de categoria não pode ser vazio para evitar wipe total do banco."
            
        logger.info(f"Apagando memórias do path: '{prefix}'")
        
        # Encontra memórias
        rows = engine.conn.execute(
            "SELECT id FROM memory WHERE path = ? OR path LIKE ?;",
            (prefix, f"{prefix}/%")
        ).fetchall()
        
        ids_to_delete = [r["id"] for r in rows]
        
        if not ids_to_delete:
            return f"Nenhuma memória encontrada sob a categoria '{prefix}'."
            
        # Deleta as memórias (CASCADE lidará com versions e references devido aos foreign keys do db.py)
        with engine.conn:
            engine.conn.executemany("DELETE FROM memory WHERE id=?;", [(i,) for i in ids_to_delete])
            
        engine.rebuild_index()
        return f"Foram apagadas {len(ids_to_delete)} memórias da categoria '{prefix}' com sucesso."
        
    except SynapseError as e:
        return f"Erro ao apagar memórias: {e}"
    except Exception as e:
        logger.error(f"Erro inesperado em forget_memory: {e}")
        return f"Erro interno ao apagar memórias: {e}"


if __name__ == "__main__":
    # Quando rodado diretamente, inicializa o servidor MCP via stdio
    mcp.run()
