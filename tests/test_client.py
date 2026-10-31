"""OpenAI-compatible backend: retries, prefix routing, structured-output request body (local fake server)."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oblig.llm.client import LLMClient, LLMConfig  # noqa: E402

SCHEMA = {"type": "object", "required": ["verdict"], "properties": {"verdict": {"enum": ["SUPPORTS", "REFUTES", "NEI"]}}}


def fake_servers(n, fail_first=0):
    seen, state = [], {"fail": fail_first}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.server.server_port, body))
            if state["fail"]:
                state["fail"] -= 1
                self.send_response(503)
                self.end_headers()
                return
            data = json.dumps({"choices": [{"message": {"content": '{"verdict": "NEI"}'}}],
                               "usage": {"prompt_tokens": 7, "completion_tokens": 2}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    servers = [HTTPServer(("127.0.0.1", 0), H) for _ in range(n)]
    for s in servers:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    return ",".join(f"http://127.0.0.1:{s.server_port}/v1" for s in servers), seen, servers


def test_retry_routing_and_structured_output():
    urls, seen, servers = fake_servers(2, fail_first=1)
    cfg = LLMConfig(backend="openai", model="m", base_url=urls,
                    extra={"chat_template_kwargs": {"enable_thinking": False}, "structured_output": True, "retries": 2})
    client = LLMClient(cfg)
    msgs = [{"role": "user", "content": "Abstracts: shared evidence"}]
    r1, r2 = client.generate_json(msgs, SCHEMA, "v"), client.generate_json(msgs, SCHEMA, "v")
    assert r1.parsed == {"verdict": "NEI"} and r1.error is None and r1.prompt_tokens == 7
    assert len(seen) == 3 and seen[1][0] == seen[2][0]           # one retry; same prefix -> same server
    body = seen[-1][1]
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert "structured_output" not in body and "retries" not in body
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    for s in servers:
        s.shutdown()


def test_exhausted_retries_are_recorded_not_raised():
    urls, seen, servers = fake_servers(1, fail_first=5)
    client = LLMClient(LLMConfig(backend="openai", model="m", base_url=urls, extra={"retries": 1}))
    r = client.generate_json([{"role": "user", "content": "x"}], SCHEMA, "v")
    assert r.parsed is None and r.error.startswith("call_error") and len(seen) == 2
    for s in servers:
        s.shutdown()
