# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import unittest
from unittest.mock import MagicMock

from trae_agent.agent.agent_basics import AgentStep, AgentStepState
from trae_agent.utils.config import Config, LongTermMemoryConfig, ModelConfig, ModelProvider
from trae_agent.utils.llm_clients.llm_basics import LLMMessage, LLMResponse
from trae_agent.utils.long_term_memory import (
    LongTermMemory,
    MemoryDocument,
    MemorySection,
)
from trae_agent.utils.memory_trigger import (
    ManualMemoryTrigger,
    PeriodicMemoryTrigger,
    create_memory_trigger,
)


class TestMemoryDocument(unittest.TestCase):
    def test_to_markdown(self):
        doc = MemoryDocument(
            task_name="Fix login bug",
            sections=[
                MemorySection(step_range="1-3", problem="Login fails", conclusion="Regex issue"),
                MemorySection(step_range="4-5", problem="Fix regex", conclusion="Used re.escape"),
            ],
            created_at="2025-01-01 00:00:00",
            step_count=5,
        )
        md = doc.to_markdown()
        self.assertIn("# Long-term Memory — Task: Fix login bug", md)
        self.assertIn("## Step 1-3", md)
        self.assertIn("**Problem**: Login fails", md)
        self.assertIn("**Conclusion**: Regex issue", md)
        self.assertIn("## Step 4-5", md)
        self.assertIn("**Problem**: Fix regex", md)
        self.assertIn("**Conclusion**: Used re.escape", md)
        self.assertIn("Generated: 2025-01-01 00:00:00 | Steps: 5", md)

    def test_from_markdown_round_trip(self):
        doc = MemoryDocument(
            task_name="Fix login bug",
            sections=[
                MemorySection(step_range="1-3", problem="Login fails", conclusion="Regex issue"),
            ],
            created_at="2025-01-01 00:00:00",
            step_count=3,
        )
        md = doc.to_markdown()
        parsed = MemoryDocument.from_markdown(md)
        self.assertEqual(parsed.task_name, "Fix login bug")
        self.assertEqual(len(parsed.sections), 1)
        self.assertEqual(parsed.sections[0].step_range, "1-3")
        self.assertEqual(parsed.sections[0].problem, "Login fails")
        self.assertEqual(parsed.sections[0].conclusion, "Regex issue")

    def test_from_markdown_unknown_task(self):
        parsed = MemoryDocument.from_markdown("# unrelated content")
        self.assertEqual(parsed.task_name, "Unknown")
        self.assertEqual(len(parsed.sections), 0)

    def test_memory_section_heading(self):
        section = MemorySection(step_range="1-5", problem="x", conclusion="y")
        self.assertEqual(section.heading(), "Step 1-5")


class TestMemoryTrigger(unittest.TestCase):
    def test_manual_trigger_never_fires(self):
        trigger = ManualMemoryTrigger()
        step = AgentStep(step_number=1, state=AgentStepState.COMPLETED)
        self.assertFalse(trigger.should_trigger(step, 1))
        self.assertFalse(trigger.should_trigger(step, 100))
        self.assertEqual(trigger.trigger_type_name(), "manual")

    def test_periodic_trigger_fires_at_interval(self):
        trigger = PeriodicMemoryTrigger(interval=5)
        step = AgentStep(step_number=5, state=AgentStepState.COMPLETED)
        self.assertTrue(trigger.should_trigger(step, 5))
        # Should not fire again at same count
        self.assertFalse(trigger.should_trigger(step, 5))

    def test_periodic_trigger_does_not_fire_between_intervals(self):
        trigger = PeriodicMemoryTrigger(interval=10)
        step = AgentStep(step_number=3, state=AgentStepState.COMPLETED)
        self.assertFalse(trigger.should_trigger(step, 3))
        self.assertFalse(trigger.should_trigger(step, 7))

    def test_periodic_trigger_multiple_firings(self):
        trigger = PeriodicMemoryTrigger(interval=3)
        step = AgentStep(step_number=3, state=AgentStepState.COMPLETED)
        self.assertTrue(trigger.should_trigger(step, 3))
        self.assertFalse(trigger.should_trigger(step, 3))
        self.assertTrue(trigger.should_trigger(step, 6))
        self.assertFalse(trigger.should_trigger(step, 6))

    def test_periodic_trigger_type_name(self):
        trigger = PeriodicMemoryTrigger(interval=5)
        self.assertEqual(trigger.trigger_type_name(), "periodic(every 5 steps)")

    def test_create_memory_trigger_manual(self):
        trigger = create_memory_trigger("manual")
        self.assertIsInstance(trigger, ManualMemoryTrigger)

    def test_create_memory_trigger_periodic(self):
        trigger = create_memory_trigger("periodic", periodic_interval=7)
        self.assertIsInstance(trigger, PeriodicMemoryTrigger)
        self.assertEqual(trigger.trigger_type_name(), "periodic(every 7 steps)")

    def test_create_memory_trigger_unknown(self):
        with self.assertRaises(ValueError):
            create_memory_trigger("unknown")


class TestLongTermMemoryConfig(unittest.TestCase):
    def test_default_values(self):
        config = LongTermMemoryConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.trigger_type, "manual")
        self.assertEqual(config.periodic_interval, 10)
        self.assertEqual(config.output_dir, "memory/")
        self.assertIsNone(config.model)

    def test_custom_values(self):
        config = LongTermMemoryConfig(
            enabled=True,
            trigger_type="periodic",
            periodic_interval=5,
            output_dir="my_memory/",
        )
        self.assertTrue(config.enabled)
        self.assertEqual(config.trigger_type, "periodic")
        self.assertEqual(config.periodic_interval, 5)
        self.assertEqual(config.output_dir, "my_memory/")

    def test_yaml_parsing(self):
        yaml_str = """
model_providers:
    openai:
        api_key: test-key
        provider: openai
models:
    default_model:
        model_provider: openai
        model: gpt-4o
        max_tokens: 4096
        temperature: 0.5
        top_p: 1
        top_k: 0
        max_retries: 5
        parallel_tool_calls: true
long_term_memory:
    enabled: true
    trigger_type: periodic
    periodic_interval: 5
    output_dir: my_memory/
agents:
    trae_agent:
        model: default_model
        max_steps: 50
        enable_lakeview: false
        tools:
            - bash
"""
        config = Config.create(config_string=yaml_str)
        self.assertIsNotNone(config.trae_agent)
        self.assertIsNotNone(config.trae_agent.long_term_memory)
        ltm = config.trae_agent.long_term_memory
        self.assertTrue(ltm.enabled)
        self.assertEqual(ltm.trigger_type, "periodic")
        self.assertEqual(ltm.periodic_interval, 5)
        self.assertEqual(ltm.output_dir, "my_memory/")

    def test_yaml_parsing_disabled(self):
        yaml_str = """
model_providers:
    openai:
        api_key: test-key
        provider: openai
models:
    default_model:
        model_provider: openai
        model: gpt-4o
        max_tokens: 4096
        temperature: 0.5
        top_p: 1
        top_k: 0
        max_retries: 5
        parallel_tool_calls: true
long_term_memory:
    enabled: false
agents:
    trae_agent:
        model: default_model
        max_steps: 50
        enable_lakeview: false
        tools:
            - bash
"""
        config = Config.create(config_string=yaml_str)
        self.assertIsNotNone(config.trae_agent.long_term_memory)
        self.assertFalse(config.trae_agent.long_term_memory.enabled)


class TestBuildMemoryMessage(unittest.TestCase):
    def test_build_message_with_sections(self):
        model = ModelConfig(
            model="gpt-4o",
            model_provider=ModelProvider(api_key="test", provider="openai"),
            max_tokens=4096,
            temperature=0.5,
            top_p=1,
            top_k=0,
            max_retries=5,
            parallel_tool_calls=True,
        )
        ltm = LongTermMemory(
            config=LongTermMemoryConfig(enabled=True),
            fallback_model=model,
        )
        ltm._sections = [
            MemorySection(step_range="1-3", problem="Login fails", conclusion="Regex issue"),
        ]
        msg = ltm.build_memory_message()
        self.assertIsNotNone(msg)
        self.assertIn("Login fails", msg.content)
        self.assertIn("Regex issue", msg.content)
        self.assertEqual(msg.role, "user")

    def test_build_message_no_sections(self):
        model = ModelConfig(
            model="gpt-4o",
            model_provider=ModelProvider(api_key="test", provider="openai"),
            max_tokens=4096,
            temperature=0.5,
            top_p=1,
            top_k=0,
            max_retries=5,
            parallel_tool_calls=True,
        )
        ltm = LongTermMemory(
            config=LongTermMemoryConfig(enabled=True),
            fallback_model=model,
        )
        msg = ltm.build_memory_message()
        self.assertIsNone(msg)


class TestInjectMemoryIntoMessages(unittest.TestCase):
    def _make_agent(self):
        from trae_agent.agent.base_agent import BaseAgent

        class ConcreteAgent(BaseAgent):
            def new_task(self, task, extra_args=None, tool_names=None):
                pass

            async def cleanup_mcp_clients(self):
                pass

        agent = ConcreteAgent.__new__(ConcreteAgent)
        agent._long_term_memory = None
        agent._memory_trigger = None
        agent._max_steps = 10
        agent._tools = []
        agent._cli_console = None
        agent._trajectory_recorder = None
        agent._tool_confirmation_config = MagicMock(enabled=False)
        agent._tool_confirmation_approved_all = False
        agent._allowed_command_prefixes = []
        agent._allowed_tool_names = set()
        return agent

    def test_inject_preserves_system_and_recent(self):
        agent = self._make_agent()
        ltm = MagicMock()
        ltm.build_memory_message.return_value = LLMMessage(
            role="user", content="Memory summary"
        )
        agent._long_term_memory = ltm

        messages = [LLMMessage(role="system", content="system prompt")]
        for i in range(25):
            messages.append(LLMMessage(role="user", content=f"msg {i}"))

        result = agent.inject_memory_into_messages(messages, keep_recent=4)
        self.assertEqual(result[0].content, "system prompt")
        self.assertEqual(result[1].content, "Memory summary")
        self.assertEqual(len(result), 6)  # system + memory + 4 recent

    def test_no_injection_without_memory(self):
        agent = self._make_agent()
        messages = [LLMMessage(role="system", content="system prompt")]
        for i in range(25):
            messages.append(LLMMessage(role="user", content=f"msg {i}"))

        result = agent.inject_memory_into_messages(messages)
        self.assertEqual(result, messages)


class TestExtractMemory(unittest.TestCase):
    def test_extract_memory_with_mock_llm(self):
        model = ModelConfig(
            model="gpt-4o",
            model_provider=ModelProvider(api_key="test", provider="openai"),
            max_tokens=4096,
            temperature=0.5,
            top_p=1,
            top_k=0,
            max_retries=5,
            parallel_tool_calls=True,
        )
        ltm = LongTermMemory(
            config=LongTermMemoryConfig(enabled=True),
            fallback_model=model,
        )
        ltm.set_task("Fix login bug")

        # Mock the LLM client
        mock_response = LLMResponse(
            content='<group steps="1-3">\n<problem>Login fails with special chars</problem>\n<conclusion>Regex in auth/validator.py:42 needs escaping</conclusion>\n</group>',
            model="gpt-4o",
        )
        ltm._llm_client = MagicMock()
        ltm._llm_client.chat = MagicMock(return_value=mock_response)

        steps = [
            AgentStep(
                step_number=1,
                state=AgentStepState.COMPLETED,
                llm_response=LLMResponse(content="Looking at login code", model="gpt-4o"),
            ),
            AgentStep(
                step_number=2,
                state=AgentStepState.COMPLETED,
                llm_response=LLMResponse(content="Found the issue", model="gpt-4o"),
            ),
        ]

        import asyncio

        doc = asyncio.run(ltm.extract_memory(steps))
        self.assertIsNotNone(doc)
        self.assertEqual(len(doc.sections), 1)
        self.assertEqual(doc.sections[0].step_range, "1-3")
        self.assertIn("Login fails", doc.sections[0].problem)
        self.assertIn("Regex", doc.sections[0].conclusion)

    def test_extract_and_save_creates_file(self):
        import asyncio
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            model = ModelConfig(
                model="gpt-4o",
                model_provider=ModelProvider(api_key="test", provider="openai"),
                max_tokens=4096,
                temperature=0.5,
                top_p=1,
                top_k=0,
                max_retries=5,
                parallel_tool_calls=True,
            )
            ltm = LongTermMemory(
                config=LongTermMemoryConfig(enabled=True, output_dir=tmpdir),
                fallback_model=model,
            )
            ltm.set_task("Test task")

            mock_response = LLMResponse(
                content='<group steps="1-2">\n<problem>Test problem</problem>\n<conclusion>Test conclusion</conclusion>\n</group>',
                model="gpt-4o",
            )
            ltm._llm_client = MagicMock()
            ltm._llm_client.chat = MagicMock(return_value=mock_response)

            steps = [
                AgentStep(
                    step_number=1,
                    state=AgentStepState.COMPLETED,
                    llm_response=LLMResponse(content="Step 1", model="gpt-4o"),
                ),
            ]

            path = asyncio.run(ltm.extract_and_save(steps))
            self.assertIsNotNone(path)
            import os

            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                content = f.read()
            self.assertIn("Test task", content)
            self.assertIn("Test problem", content)
            self.assertIn("Test conclusion", content)


if __name__ == "__main__":
    unittest.main()
