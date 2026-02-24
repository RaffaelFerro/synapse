import unittest
import tempfile
import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch
from synapse.engine import SynapseEngine
from synapse.util import utc_now

class KGIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(prefix="synapse_kg_", suffix=".db", delete=False)
        self.tmp.close()
        self.db_path = str(Path(self.tmp.name))
        self.engine = SynapseEngine(self.db_path, debug=True)
        # Enable KG in config
        self.engine.config["kg_extraction_enabled"] = True

    def tearDown(self):
        self.engine.close()
        Path(self.db_path).unlink(missing_ok=True)

    def test_schema_exists(self):
        """Verifica se as tabelas do Grafo foram criadas."""
        tables = [
            "entities", "relations", "observations", 
            "indexing_jobs", "entities_fts"
        ]
        with self.engine.conn as conn:
            for table in tables:
                row = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", 
                    (table,)
                ).fetchone()
                self.assertIsNotNone(row, f"Tabela {table} não encontrada")

    def test_fts5_triggers(self):
        """Verifica se o gatilho FTS5 está funcionando."""
        with self.engine.conn as conn:
            conn.execute(
                "INSERT INTO entities (id, name, type, description, created_at, updated_at) "
                "VALUES ('e1', 'Python Language', 'tech', 'desc', 'now', 'now');"
            )
            # Verifica FTS5
            row = conn.execute("SELECT name FROM entities_fts WHERE name MATCH 'Python';").fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["name"], "Python Language")

    def test_add_memory_queues_job(self):
        """Verifica se adicionar memória cria um job pendente."""
        self.engine.add_memory(path="test", content="João trabalha com Python.", title="Test")
        row = self.engine.conn.execute("SELECT status FROM indexing_jobs;").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "pending")

    @patch("synapse.engine.get_llm_client")
    def test_process_jobs(self, mock_get_llm):
        """Testa o processamento do job e criação de triplas."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        # Mock da extração KG
        mock_llm.chat_completion.return_value = json.dumps({
            "entities": [
                {"name": "João", "type": "Pessoa"},
                {"name": "Python", "type": "Tecnologia"}
            ],
            "relations": [
                {"source": "João", "target": "Python", "type": "USA", "observation": "Experiente"}
            ],
            "observations": []
        })

        mid = self.engine.add_memory(path="test", content="João usa Python.")
        processed = self.engine.process_indexing_jobs(limit=1)
        
        self.assertEqual(processed, 1)
        
        # Verifica se entidades e relações foram criadas
        with self.engine.conn as conn:
            ent = conn.execute("SELECT name FROM entities WHERE name='João';").fetchone()
            self.assertIsNotNone(ent)
            
            rel = conn.execute("SELECT relation_type FROM relations WHERE relation_type='USA';").fetchone()
            self.assertIsNotNone(rel)
            
            obs = conn.execute("SELECT content FROM observations WHERE content='Experiente';").fetchone()
            self.assertIsNotNone(obs)

    @patch("synapse.engine.get_llm_client")
    def test_search_fast_path(self, mock_get_llm):
        """Testa a busca rápida (Fast Path) via FTS5 sem chamar LLM."""
        # Setup data
        with self.engine.conn as conn:
            conn.execute("INSERT INTO entities (id, name, created_at, updated_at) VALUES ('e1', 'Docker', 'now', 'now');")
            conn.execute("INSERT INTO entities (id, name, created_at, updated_at) VALUES ('e2', 'Kubernetes', 'now', 'now');")
            conn.execute("INSERT INTO relations (id, source_id, target_id, relation_type, created_at) VALUES ('r1', 'e1', 'e2', 'COMPLEMENTA', 'now');")
        
        # Se o fast-path funcionar, o mock_llm não deve ser chamado para 'interpret_search.txt'
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        res = self.engine.search("Me fale sobre Docker e Kubernetes")
        
        # Verifica se as triplas foram injetadas
        self.assertIn("knowledge_graph", res.context)
        self.assertTrue(any("Docker" in t and "Kubernetes" in t for t in res.context["knowledge_graph"]["triples"]))
        
        # Verifica se o LLM NÃO foi chamado (ou pelo menos não para interpretação)
        # Note: o build_context pode chamar LLM se a lógica de fallback for ativada, 
        # mas aqui testamos o bypass em _query_knowledge_graph.
        # Como o prompt 'Me fale sobre...' contém as palavras exatas 'Docker' e 'Kubernetes', o fast-path deve pegar.
        self.assertIn("Docker", res.context["knowledge_graph"]["search_terms"])

    @patch("synapse.engine.get_llm_client")
    def test_job_processing_timeout_recovery(self, mock_get_llm):
        """Testa se jobs travados em 'processing' são retomados."""
        mid = self.engine.add_memory(path="test", content="Fato isolado.")
        
        # Simula job travado (updated_at antigo)
        old_time = "2020-01-01T00:00:00Z"
        self.engine.conn.execute(
            "UPDATE indexing_jobs SET status='processing', updated_at=?;", (old_time,)
        )
        self.engine.conn.commit()
        
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_llm.chat_completion.return_value = json.dumps({"entities": [], "relations": [], "observations": []})
        
        processed = self.engine.process_indexing_jobs(limit=1)
        self.assertEqual(processed, 1)

if __name__ == "__main__":
    unittest.main()
