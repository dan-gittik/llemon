from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar

import llemon.sync as llemon
from llemon.sync.types import NS, Error, FilesArgument, HistoryArgument, RenderArgument, ToolsArgument
from llemon.utils import filtered_dict, schema_to_model, trim

if TYPE_CHECKING:
    from llemon.sync import LLM, GenerateObjectRequest
    from llemon.sync.serializeable import DumpRefs, LoadRefs, Unpacker


class ClassifyRequest(llemon.GenerateRequest):

    BOOLEAN_ANSWERS: ClassVar[list[str]] = ["false", "true"]
    NULL_ANSWER: ClassVar[str] = "none of the above"

    def __init__(
        self,
        *,
        llm: LLM,
        question: str,
        answers: list[str] | type[bool],
        user_input: str,
        reasoning: bool = False,
        null_answer: bool = True,
        context: NS | None = None,
        render: RenderArgument = None,
        history: HistoryArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        cache: bool | None = None,
        timeout: float | None = None,
        **provider_options: Any,
    ) -> None:
        super().__init__(
            llm=llm,
            user_input=user_input,
            context=context,
            render=render,
            history=history,
            files=files,
            tools=tools,
            use_tool=use_tool,
            temperature=0.0,
            seed=0,
            cache=cache,
            timeout=timeout,
            **provider_options,
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

    def __str__(self) -> str:
        question = f"{self.question!r} [{'/'.join(self.answers)}]"
        return f"{self.llm}.classify({question!r}, {self.user_input!r})"

    @cached_property
    def use_logit_biasing(self) -> bool:
        return bool(self.llm.config.supports_logit_biasing and not self.reasoning)

    def check_supported(self) -> None:
        super().check_supported()
        if self.reasoning and not self.llm.config.supports_objects:
            raise Error(f"{self.llm} doesn't support reasoning in classification")

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
            llm=self.llm,
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
            cache=self.cache,
        )

    @classmethod
    def _restore(cls, unpacker: Unpacker, refs: LoadRefs) -> tuple[NS, NS]:
        args, attrs = super()._restore(unpacker, refs)
        args.update(
            question=unpacker.get("question", str),
            answers=unpacker.get("answers", list),
            reasoning=unpacker.get("reasoning", bool, None),
        )
        return args, attrs

    def _dump(self, refs: DumpRefs) -> NS:
        data = super()._dump(refs)
        data.update(
            filtered_dict(
                question=self.question,
                answers=self.answers,
                reasoning=self.reasoning,
            )
        )
        return data

    def _prepare_instructions(self) -> str:
        if self.reasoning:
            appendix = "Add the reasoning behind your decision."
        else:
            appendix = "Do not add any other text, formatting or punctuation."
        return trim(
            f"""
        You are an expert classifier.
        Given a classification task in the form of a question, and a list of labels in the form of numbered answers,
        respond to each user input by answering the question about it with the most appropriate answer's NUMBER ONLY.
        {appendix}

        # Question
        {self.question}

        # Answers
        {"\n".join(f"{i}. {answer}" for i, answer in enumerate(self.answers))}
        """
        )


class ClassifyResponse(llemon.GenerateResponse):

    request: ClassifyRequest

    def __init__(self, request: ClassifyRequest) -> None:
        super().__init__(request)
        self.answer: str | None = None
        self.reasoning: str | None = None

    def __str__(self) -> str:
        return f"{self.request.llm}: {self.answer}"

    def complete_answer(self, answer: str, reasoning: str | None = None) -> None:
        self.answer = answer
        self.reasoning = reasoning
        text = answer
        if reasoning:
            text = f"{text} ({reasoning})"
        self.complete_text(text)

    def _restore(self, unpacker: Unpacker, refs: LoadRefs) -> None:
        super()._restore(unpacker, refs)
        self.answer = unpacker.get("texts", list)[0]
        self.reasoning = unpacker.get("reasoning", str, None)

    def _dump(self, refs: DumpRefs) -> NS:
        data = super()._dump(refs)
        data.update(
            filtered_dict(
                reasoning=self.reasoning,
            )
        )
        return data
