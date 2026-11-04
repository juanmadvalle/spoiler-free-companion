"""OpenAI-compatible provider adapter, tested against a fake HTTP function (no network, no key)."""
import json
import os
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from companion import answer, llm
from companion.agent import Session, run_agent
from companion.llm import OpenAICompatClient, convert_messages, parse_message, settings
from companion.store import Library
from tests.test_companion import CHAPTERS

CONFIG = NS(key="k", url="http://example.test/v1", model="m")


class FakeHttp:
    def __init__(self, *messages):
        self.messages, self.calls = list(messages), []

    def __call__(self, url, headers, payload):
        self.calls.append((url, headers, payload))
        if url.endswith("/models"):
            return {"data": [{"id": "models/b-model"}, {"id": "models/a-model"}]}
        return {"choices": [{"message": self.messages.pop(0)}]}


def tool_call(name, arguments, call_id="c1"):
    return {"content": None, "tool_calls": [{"id": call_id, "type": "function",
                                             "function": {"name": name, "arguments": json.dumps(arguments)}}]}


ISOLATED = {"COMPANION_CONFIG": os.path.join(tempfile.gettempdir(), "companion-test-missing", "config.json"),
            "COMPANION_NO_KEYRING": "1"}  # never read the developer's real stored key


class SettingsTests(unittest.TestCase):
    def test_settings_come_from_environment_only(self):
        with patch.dict(os.environ, {**ISOLATED}, clear=True):
            self.assertIsNone(settings())
        with patch.dict(os.environ, {**ISOLATED, "GEMINI_API_KEY": "abc"}, clear=True):
            config = settings()
            self.assertEqual((config.key, config.url), ("abc", llm.GEMINI_URL))
        with patch.dict(os.environ, {**ISOLATED, "COMPANION_LLM_BASE_URL": "http://localhost:11434/v1/",
                                     "COMPANION_LLM_MODEL": "llama"}, clear=True):
            config = settings()
            self.assertEqual((config.key, config.url, config.model), ("", "http://localhost:11434/v1", "llama"))

    def test_models_listing_strips_prefix_and_sorts(self):
        self.assertEqual(llm.list_models(CONFIG, FakeHttp()), ["a-model", "b-model"])


class RetryTests(unittest.TestCase):
    def _http_error(self, code):
        import io
        import urllib.error

        return urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO(b'{"error": "busy"}'))

    def test_busy_server_is_retried_then_succeeds(self):
        import io

        calls, sleeps = [], []

        def urlopen(request, timeout=None):
            calls.append(1)
            if len(calls) < 3:
                raise self._http_error(503)
            return io.BytesIO(b'{"ok": true}')

        with patch("urllib.request.urlopen", urlopen):
            self.assertEqual(llm.post("http://x", {}, {"a": 1}, sleep=sleeps.append), {"ok": True})
        self.assertEqual((len(calls), sleeps), (3, [3, 8]))

    def test_non_retryable_errors_fail_immediately_and_persistent_ones_give_up(self):
        calls = []

        def forbidden(request, timeout=None):
            calls.append(1)
            raise self._http_error(400)

        with patch("urllib.request.urlopen", forbidden), self.assertRaises(RuntimeError):
            llm.post("http://x", {}, {}, sleep=lambda s: None)
        self.assertEqual(len(calls), 1)
        calls.clear()

        def busy(request, timeout=None):
            calls.append(1)
            raise self._http_error(429)

        with patch("urllib.request.urlopen", busy), self.assertRaisesRegex(RuntimeError, "rate limit"):
            llm.post("http://x", {}, {}, sleep=lambda s: None)
        self.assertEqual(len(calls), 4)

    def test_timeouts_are_retried_twice(self):
        calls = []

        def slow(request, timeout=None):
            calls.append(1)
            raise TimeoutError("read timed out")

        with patch("urllib.request.urlopen", slow), self.assertRaisesRegex(RuntimeError, "in time"):
            llm.post("http://x", {}, {}, sleep=lambda s: None)
        self.assertEqual(len(calls), 3)


class AdapterTests(unittest.TestCase):
    def test_messages_round_trip_between_formats(self):
        blocks = parse_message(tool_call("search_passages", {"query": "crystal"}))
        self.assertEqual((blocks[0].type, blocks[0].name, blocks[0].input), ("tool_use", "search_passages", {"query": "crystal"}))
        converted = convert_messages([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [NS(type="text", text="thinking"), *blocks]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "result"}]}])
        self.assertEqual([m["role"] for m in converted], ["user", "assistant", "tool"])
        self.assertEqual(converted[1]["tool_calls"][0]["function"]["name"], "search_passages")
        self.assertEqual(converted[2], {"role": "tool", "tool_call_id": "c1", "content": "result"})

    def test_malformed_tool_arguments_do_not_crash(self):
        message = {"content": "", "tool_calls": [{"id": "x", "function": {"name": "n", "arguments": "{not json"}}]}
        self.assertEqual(parse_message(message)[0].input, {})

    def test_agent_loop_runs_through_the_adapter_and_cannot_pass_the_horizon(self):
        library = Library(":memory:")
        self.addCleanup(library.close)
        library.add_work("S", CHAPTERS)
        http = FakeHttp(tool_call("read_chapter", {"chapter": 4, "horizon": 10**6}),
                        {"content": "Marlowe found a crystal [3]."})
        client = OpenAICompatClient(CONFIG, http)
        session = Session(library, "S", library.horizon_after_chapter("S", 1))
        self.assertEqual(run_agent(session, "What did Marlowe find?", client=client), "Marlowe found a crystal [3].")
        second_request = http.calls[1][2]
        tool_message = next(m for m in second_request["messages"] if m["role"] == "tool")
        self.assertNotIn("delta", tool_message["content"])
        self.assertNotIn("charlie", tool_message["content"])
        tools = http.calls[0][2]["tools"]
        self.assertTrue(all("horizon" not in t["function"]["parameters"]["properties"] for t in tools))

    def test_non_agent_answerer_uses_only_the_given_passages(self):
        http = FakeHttp({"content": "ok"})
        library = Library(":memory:")
        self.addCleanup(library.close)
        library.add_work("S", CHAPTERS)
        horizon = library.horizon_after_chapter("S", 1)
        with patch.object(llm, "post", http), patch.dict(os.environ, {**ISOLATED, "GEMINI_API_KEY": "abc"}, clear=True):
            text, _ = answer.ask(library, "S", "tell me about the deltaite crystal", horizon, answer.default_answerer())
        self.assertEqual(text, "ok")
        sent = json.dumps(http.calls[0][2])
        self.assertNotIn("delta", sent.replace("deltaite crystal", ""))  # only the question itself mentions it
        self.assertNotIn("charlie", sent)

    def test_gemini_thought_signature_is_echoed_back_with_the_tool_call(self):
        message = tool_call("search_passages", {"query": "x"})
        message["tool_calls"][0]["extra_content"] = {"google": {"thought_signature": "sig-123"}}
        blocks = parse_message(message)
        converted = convert_messages([{"role": "assistant", "content": blocks}])
        self.assertEqual(converted[0]["tool_calls"][0]["extra_content"]["google"]["thought_signature"], "sig-123")

    def test_unconfigured_provider_gives_a_clear_error(self):
        with patch.dict(os.environ, {**ISOLATED}, clear=True):
            with self.assertRaises(RuntimeError):
                OpenAICompatClient()


if __name__ == "__main__":
    unittest.main()
