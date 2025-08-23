from __future__ import annotations

from functools import cached_property
from typing import ClassVar

from llemon.sync.llm_model import LLMModel
from llemon.errors import ConfigurationError
from llemon.sync.generate import GenerateRequest, GenerateResponse
from llemon.sync.generate_object import GenerateObjectRequest
from llemon.sync.types import NS, History, RenderArgument, FilesArgument, ToolsArgument
from llemon.utils.trim import trim
from llemon.utils.schema import schema_to_model


class ClassifyRequest(GenerateRequest):

    BOOLEAN_ANSWERS: ClassVar[list[str]] = ["false", "true"]
    NULL_ANSWER: ClassVar[str] = "none of the above"

    def __init__(
        self,
        *,
        model: LLMModel,
        history: History | None = None,
        question: str,
        answers: list[str] | type[bool],
        user_input: str | None = None,
        reasoning: bool = False,
        null_answer: bool = True,
        context: NS | None = None,
        render: RenderArgument | None = None,
        files: FilesArgument = None,
        tools: ToolsArgument | None = None,
        use_tool: bool | str | None = None,
    ) -> None:
        super().__init__(
            model=model,
            history=history,
            user_input=user_input,
            context=context,
            render=render,
            files=files,
            tools=tools,
            use_tool=use_tool,
            # temperature=0.0,
            seed=0,
        )
        if answers is bool:
            answers = self.BOOLEAN_ANSWERS
        else:
            assert isinstance(answers, list)
            if null_answer:
                answers.append(self.NULL_ANSWER)
        self.question = question
        self.answers = answers
        self.reasoning = reasoning
        self.null_answer = null_answer
        if self.use_logit_biasing:
            self.max_tokens = 1
            self.return_incomplete_message = True
        self.instructions = self._prepare_instructions()

    @cached_property
    def use_logit_biasing(self) -> bool:
        return bool(self.model.config.supports_logit_biasing and not self.reasoning)

    def check_supported(self) -> None:
        super().check_supported()
        if self.reasoning and not self.model.config.supports_json:
            raise ConfigurationError(f"{self.model} doesn't support reasoning in classification")
    
    def to_object_request(self) -> GenerateObjectRequest:
        properties = {
            "answer": {
                "type": "integer",
                "maximum": len(self.answers) - 1,
            },
        }
        if self.reasoning:
            properties["reasoning"] = {
                "type": "string",
            }
        schema = {
            "title": "classification",
            "type": "object",
            "properties": properties,
        }
        return GenerateObjectRequest(
            model=self.model,
            schema=schema_to_model(schema),
            history=self.history,
            instructions=self._prepare_instructions(),
            user_input=self.user_input,
            context=self.context,
            render=self.rendering,
            files=self.files,
            tools=self.tools,
            use_tool=self.use_tool,
            temperature=self.temperature,
            seed=self.seed,
        )

    def _prepare_instructions(self) -> str:
        if self.reasoning:
            appendix = "Add the reasoning behind your decision."
        else:
            appendix = "Do not add any other text, formatting or punctuation."
        return trim(f"""
        You are an expert classifier.
        Given a classification task in the form of a question, and a list of labels in the form of numbered answers,
        respond to each user input by answering the question about it with the most appropriate answer's NUMBER ONLY.
        {appendix}

        # Question
        {self.question}

        # Answers
        {"\n".join(f"{i}. {answer}" for i, answer in enumerate(self.answers))}
        """)


class ClassifyResponse(GenerateResponse):

    request: ClassifyRequest

    def __init__(self, request: ClassifyRequest) -> None:
        super().__init__(request)
        self.answer: str | None = None
        self.reasoning: str | None = None

    def __str__(self) -> str:
        return f"{self.request.model}: {self.answer}"

    def dump(self) -> NS:
        data = super().dump()
        data.update(
            answer=self.answer,
        )
        if self.reasoning:
            data["reasoning"] = self.reasoning
        return data

    def complete_answer(self, answer: str, reasoning: str | None = None) -> None:
        self.answer = answer
        self.reasoning = reasoning
        text = answer
        if reasoning:
            text = f"{text} ({reasoning})"
        super().complete_text(text)

    @classmethod
    def _restore(self, data: NS) -> tuple[NS, NS]:
        args, attrs = super()._restore(data)
        attrs.update(
            answer=data["answer"],
            reasoning=data.get("reasoning"),
        )
        return args, attrs
