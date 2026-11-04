"""Agent loop with a scripted fake model: the harness, not the model, bounds what it can read."""
import unittest
from types import SimpleNamespace as NS

from companion.agent import TOOLS, Session, run_agent
from companion.store import Library
from tests.test_companion import CHAPTERS


def text(value):
    return NS(type="text", text=value)


def call(name, **arguments):
    return NS(type="tool_use", id=f"id-{name}", name=name, input=arguments)


class ScriptedClient:
    """Replays model turns and records every tool result the harness sent back."""

    def __init__(self, turns):
        self.turns, self.tool_results, self.messages = list(turns), [], self
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        for message in kwargs["messages"]:
            if message["role"] == "user" and isinstance(message["content"], list):
                self.tool_results = [r["content"] for m in kwargs["messages"]
                                     if m["role"] == "user" and isinstance(m["content"], list) for r in m["content"]]
        return NS(content=self.turns.pop(0))


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.library = Library(":memory:")
        self.library.add_work("Synthetic", CHAPTERS)
        self.addCleanup(self.library.close)
        self.session = Session(self.library, "Synthetic", self.library.horizon_after_chapter("Synthetic", 2))

    def test_no_tool_exposes_horizon_book_or_position(self):
        forbidden = {"horizon", "title", "book", "work", "ordinal", "position", "max_chapter"}
        for tool in TOOLS:
            self.assertFalse(forbidden & set(tool["input_schema"]["properties"]), tool["name"])

    def test_model_cannot_read_past_the_horizon_by_any_argument(self):
        attempts = [
            ("read_chapter", {"chapter": 3}),
            ("read_chapter", {"chapter": 4, "horizon": 10**6}),
            ("search_passages", {"query": "charlieite deltaite", "horizon": 10**6, "limit": 100}),
            ("read_chapter", {"chapter": "4; DROP TABLE chunks"}),
            ("read_chapter", {"chapter": 2, "offset": 10**6}),
            ("nope", {"horizon": 10**6}),
        ]
        for name, arguments in attempts:
            result = self.session.run_tool(name, arguments)
            self.assertNotIn("charlie", result)
            self.assertNotIn("delta", result)

    def test_read_chapter_inside_horizon_works(self):
        result = self.session.run_tool("read_chapter", {"chapter": 2})
        self.assertIn("bravoite", result)

    def test_full_loop_returns_answer_after_tool_use(self):
        client = ScriptedClient([
            [call("search_passages", query="crystal", horizon=10**6)],
            [call("read_chapter", chapter=4)],
            [text("Marlowe found a crystal [3].")],
        ])
        answer = run_agent(self.session, "What did Marlowe find?", client=client)
        self.assertEqual(answer, "Marlowe found a crystal [3].")
        for result in client.tool_results:
            self.assertNotIn("charlie", result)
            self.assertNotIn("delta", result)
        self.assertEqual(len(self.session.log), 2)

    def test_tools_sent_to_model_have_no_horizon_parameter(self):
        client = ScriptedClient([[text("done")]])
        run_agent(self.session, "hi", client=client)
        self.assertEqual(client.requests[0]["tools"], TOOLS)

    def test_ask_user_ends_the_turn_with_the_question(self):
        client = ScriptedClient([[call("ask_user", question="Which Ines do you mean?")]])
        self.assertEqual(run_agent(self.session, "Who is she?", client=client), "Which Ines do you mean?")

    def test_step_limit(self):
        client = ScriptedClient([[call("recent_passages")]] * 3)
        self.assertIn("step limit", run_agent(self.session, "loop", client=client, max_turns=3))


if __name__ == "__main__":
    unittest.main()
