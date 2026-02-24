from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from synapse.engine import SynapseEngine, SynapseError
from synapse.tokenize import join_terms


class EngineTests(unittest.TestCase):
    def _engine(self) -> tuple[SynapseEngine, Path]:
        tmp = tempfile.NamedTemporaryFile(prefix="synapse_", suffix=".db", delete=False)
        tmp.close()
        path = Path(tmp.name)
        eng = SynapseEngine(str(path), debug=True)
        eng.config["storage_limit_bytes"] = 0
        return eng, path

    def test_add_and_search(self) -> None:
        eng, path = self._engine()
        try:
            mid = eng.add_memory(path="projeto/decisoes", title="DB", content="Usar SQLite.", rebuild_index=True)
            res = eng.search("sqlite")
            ids = [m["id"] for m in res.context["memories"] if not m.get("missing")]
            self.assertIn(mid, ids)
        finally:
            eng.close()
            path.unlink(missing_ok=True)

    def test_split_creates_parts(self) -> None:
        eng, path = self._engine()
        try:
            content = join_terms(["palavra"] * 600)
            parent_id = eng.add_memory(path="teste/grande", title="Grande", content=content, rebuild_index=True)
            parent = eng.show_memory(parent_id)
            self.assertEqual(parent["part_total"], 2)

            rows = eng.conn.execute(
                "SELECT token_count FROM memory WHERE parent_id=? ORDER BY part_index ASC;", (parent_id,)
            ).fetchall()
            self.assertEqual(len(rows), 2)
            self.assertEqual(int(rows[0]["token_count"]), 512)
            self.assertEqual(int(rows[1]["token_count"]), 88)
        finally:
            eng.close()
            path.unlink(missing_ok=True)

    def test_version_limit(self) -> None:
        eng, path = self._engine()
        try:
            mid = eng.add_memory(path="p/a", title="t", content="v1", rebuild_index=True)
            for i in range(1, 20):
                eng.update_memory(mid, content=f"v{i+1}")

            rows = eng.conn.execute(
                "SELECT version FROM memory_version WHERE memory_id=? ORDER BY version ASC;", (mid,)
            ).fetchall()
            self.assertLessEqual(len(rows), int(eng.config["max_versions"]))
        finally:
            eng.close()
            path.unlink(missing_ok=True)

    def test_anonymous_blocks_writes(self) -> None:
        eng, path = self._engine()
        try:
            eng.set_anonymous(True)
            with self.assertRaises(SynapseError):
                eng.add_memory(path="x/y", content="teste", rebuild_index=True)
            _ = eng.search("teste")  # read-only ok
        finally:
            eng.close()
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

