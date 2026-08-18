"""
This file defines a useful high-level abstraction to build Gradio chatbots: ChatInterface.
Adapted for Gradio 3.x, text-only mode, without gradio.events.on
"""

from __future__ import annotations

import inspect
from typing import AsyncGenerator, Callable

import anyio
from gradio_client import utils as client_utils
from gradio_client.documentation import document, set_documentation_group

from gradio.blocks import Blocks
from gradio.components import (
    Button,
    Chatbot,
    IOComponent,
    Markdown,
    State,
    Textbox,
    get_component_instance,
)
from gradio.events import Dependency, EventListenerMethod
from gradio.helpers import create_examples as Examples  # noqa: N812
from gradio.layouts import Accordion, Column, Group, Row
from gradio.themes import ThemeClass as Theme
from gradio.utils import SyncToAsyncIterator, async_iteration
from functools import wraps

def async_lambda(f: Callable) -> Callable:
    """Turn a function into an async function.
    Useful for internal event handlers defined as lambda functions used in the codebase
    """

    @wraps(f)
    async def function_wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return function_wrapper

@document()
class ChatInterface(Blocks):
    """
    ChatInterface is Gradio's high-level abstraction for creating chatbot UIs...
    """

    def __init__(
        self,
        fn: Callable,
        post_fn: Callable,
        pre_fn: Callable,
        chatbot: Chatbot,
        *,
        post_fn_kwargs: dict = None,
        pre_fn_kwargs: dict = None,
        textbox: Textbox | None = None,
        additional_inputs: str | Component | list[str | Component] | None = None,
        additional_inputs_accordion_name: str | None = None,
        additional_inputs_accordion: str | Accordion | None = None,
        examples: Dataset = None,
        title: str | None = None,
        description: str | None = None,
        theme: Theme | str | None = None,
        css: str | None = None,
        analytics_enabled: bool | None = None,
        submit_btn: str | None | Button = "Submit",
        stop_btn: str | None | Button = "Stop",
        retry_btn: str | None | Button = "🔄  Retry",
        undo_btn: str | None | Button = "↩️ Undo",
        clear_btn: str | None | Button = "🗑️  Clear",
        autofocus: bool = True,
    ):
        super().__init__(
            analytics_enabled=analytics_enabled,
            css=css,
            title=title or "Gradio",
            theme=theme,
        )

        if post_fn_kwargs is None:
            post_fn_kwargs = []

        self.post_fn = post_fn
        self.post_fn_kwargs = post_fn_kwargs

        self.pre_fn = pre_fn
        self.pre_fn_kwargs = pre_fn_kwargs

        self.interrupter = State(None)

        self.fn = fn
        self.is_async = inspect.iscoroutinefunction(
            self.fn
        ) or inspect.isasyncgenfunction(self.fn)
        self.is_generator = inspect.isgeneratorfunction(
            self.fn
        ) or inspect.isasyncgenfunction(self.fn)

        if additional_inputs:
            if not isinstance(additional_inputs, list):
                additional_inputs = [additional_inputs]
            self.additional_inputs = [
                get_component_instance(i)
                for i in additional_inputs
            ]
        else:
            self.additional_inputs = []

        if additional_inputs_accordion_name is not None:
            print(
                "The `additional_inputs_accordion_name` parameter is deprecated and will be removed in a future version of Gradio. Use the `additional_inputs_accordion` parameter instead."
            )
            self.additional_inputs_accordion_params = {
                "label": additional_inputs_accordion_name
            }
        if additional_inputs_accordion is None:
            self.additional_inputs_accordion_params = {
                "label": "Additional Inputs",
                "open": False,
            }
        elif isinstance(additional_inputs_accordion, str):
            self.additional_inputs_accordion_params = {
                "label": additional_inputs_accordion
            }
        elif isinstance(additional_inputs_accordion, Accordion):
            self.additional_inputs_accordion_params = {
                "label": additional_inputs_accordion.label,
                "open": additional_inputs_accordion.open,
            }
        else:
            raise ValueError(
                f"The `additional_inputs_accordion` parameter must be a string or gr.Accordion, not {type(additional_inputs_accordion)}"
            )

        with self:
            if title:
                Markdown(
                    f"<h1 style='text-align: center; margin-bottom: 1rem'>{self.title}</h1>"
                )
            if description:
                Markdown(description)

            self.chatbot = chatbot

            self.buttons = [retry_btn, undo_btn, clear_btn]

            with Group():
                with Row():
                    if textbox:
                        textbox.container = False
                        textbox.show_label = False
                        textbox_ = textbox
                        if not isinstance(textbox_, Textbox):
                            raise TypeError(
                                f"Expected a gr.Textbox component, but got {type(textbox_)}"
                            )
                        self.textbox = textbox_
                    else:
                        self.textbox = Textbox(
                            container=False,
                            show_label=False,
                            label="Message",
                            placeholder="Type a message...",
                            scale=7,
                            autofocus=autofocus,
                        )
                    if submit_btn is not None:
                        if isinstance(submit_btn, Button):
                            pass
                        elif isinstance(submit_btn, str):
                            submit_btn = Button(
                                submit_btn,
                                variant="primary",
                                scale=1,
                                min_width=150,
                            )
                        else:
                            raise ValueError(
                                f"The submit_btn parameter must be a gr.Button, string, or None, not {type(submit_btn)}"
                            )
                    if stop_btn is not None:
                        if isinstance(stop_btn, Button):
                            stop_btn.visible = False
                        elif isinstance(stop_btn, str):
                            stop_btn = Button(
                                stop_btn,
                                variant="stop",
                                visible=False,
                                scale=1,
                                min_width=150,
                            )
                        else:
                            raise ValueError(
                                f"The stop_btn parameter must be a gr.Button, string, or None, not {type(stop_btn)}"
                            )
                    self.buttons.extend([submit_btn, stop_btn])

                self.fake_api_btn = Button("Fake API", visible=False)
                self.fake_response_textbox = Textbox(label="Response", visible=False)
                (
                    self.retry_btn,
                    self.undo_btn,
                    self.clear_btn,
                    self.submit_btn,
                    self.stop_btn,
                ) = self.buttons

            any_unrendered_inputs = any(
                not inp.is_rendered for inp in self.additional_inputs
            )
            if self.additional_inputs and any_unrendered_inputs:
                with Accordion(**self.additional_inputs_accordion_params):
                    for input_component in self.additional_inputs:
                        if not input_component.is_rendered:
                            input_component.render()

            self.saved_input = State()
            self.chatbot_state = (
                State(self.chatbot.value) if self.chatbot.value else State([])
            )

            self._setup_events()
            self._setup_api()

        if examples:
            examples.click(lambda x: x[0], inputs=[examples], outputs=self.textbox, show_progress=False, queue=False)

    def _create_submit_chain(self, trigger) -> Dependency:
        """Создает цепочку событий для отправки сообщения"""
        submit_fn = self._stream_fn if self.is_generator else self._submit_fn
        
        return (
            trigger(
                self._clear_and_save_textbox,
                [self.textbox],
                [self.textbox, self.saved_input],
                api_name=False,
                queue=False,
            )
            .then(
                self.pre_fn,
                **self.pre_fn_kwargs,
                api_name=False,
                queue=False,
            )
            .then(
                self._display_input,
                [self.saved_input, self.chatbot_state],
                [self.chatbot, self.chatbot_state],
                api_name=False,
                queue=False,
            )
            .then(
                submit_fn,
                [self.saved_input, self.chatbot_state] + self.additional_inputs,
                [self.chatbot, self.chatbot_state, self.interrupter],
                api_name=False,
            )
            .then(
                self.post_fn,
                **self.post_fn_kwargs,
                api_name=False,
            )
        )

    def _setup_events(self) -> None:
        # Создаем цепочку для textbox.submit
        submit_triggers = [self.textbox.submit]
        submit_event_textbox = self._create_submit_chain(self.textbox.submit)
        
        # Создаем цепочку для submit_btn.click, если кнопка есть
        if self.submit_btn:
            submit_event_btn = self._create_submit_chain(self.submit_btn.click)
            submit_triggers.append(self.submit_btn.click)
            # Сохраняем обе цепочки для возможности отмены
            self._submit_events = [submit_event_textbox, submit_event_btn]
        else:
            self._submit_events = [submit_event_textbox]
        
        self._setup_stop_events(submit_triggers, self._submit_events)

        if self.retry_btn:
            retry_fn = self._stream_fn if self.is_generator else self._submit_fn
            retry_event = (
                self.retry_btn.click(
                    self._delete_prev_fn,
                    [self.saved_input, self.chatbot_state],
                    [self.chatbot, self.saved_input, self.chatbot_state],
                    api_name=False,
                    queue=False,
                )
                .then(
                    self.pre_fn,
                    **self.pre_fn_kwargs,
                    api_name=False,
                    queue=False,
                )
                .then(
                    self._display_input,
                    [self.saved_input, self.chatbot_state],
                    [self.chatbot, self.chatbot_state],
                    api_name=False,
                    queue=False,
                )
                .then(
                    retry_fn,
                    [self.saved_input, self.chatbot_state] + self.additional_inputs,
                    [self.chatbot, self.chatbot_state],
                    api_name=False,
                )
                .then(
                    self.post_fn,
                    **self.post_fn_kwargs,
                    api_name=False,
                )
            )
            self._setup_stop_events([self.retry_btn.click], [retry_event])

        if self.undo_btn:
            self.undo_btn.click(
                self._delete_prev_fn,
                [self.saved_input, self.chatbot_state],
                [self.chatbot, self.saved_input, self.chatbot_state],
                api_name=False,
                queue=False,
            ).then(
                self.pre_fn,
                **self.pre_fn_kwargs,
                api_name=False,
                queue=False,
            ).then(
                async_lambda(lambda x: x),
                [self.saved_input],
                [self.textbox],
                api_name=False,
                queue=False,
            ).then(
                self.post_fn,
                **self.post_fn_kwargs,
                api_name=False,
            )

        if self.clear_btn:
            self.clear_btn.click(
                async_lambda(lambda: ([], [], None)),
                None,
                [self.chatbot, self.chatbot_state, self.saved_input],
                queue=False,
                api_name=False,
            ).then(
                self.pre_fn,
                **self.pre_fn_kwargs,
                api_name=False,
                queue=False,
            ).then(
                self.post_fn,
                **self.post_fn_kwargs,
                api_name=False,
            )

    def _setup_stop_events(
        self, event_triggers: list[Callable], events_to_cancel: list[Dependency]
    ) -> None:
        def perform_interrupt(ipc):
            if ipc is not None:
                ipc()
            return

        if self.stop_btn and self.is_generator:
            if self.submit_btn:
                for event_trigger in event_triggers:
                    event_trigger(
                        async_lambda(
                            lambda: (
                                Button.update(visible=False),
                                Button.update(visible=True),
                            )
                        ),
                        None,
                        [self.submit_btn, self.stop_btn],
                        api_name=False,
                        queue=False,
                    )
                for event_to_cancel in events_to_cancel:
                    event_to_cancel.then(
                        async_lambda(lambda: (Button.update(visible=True), Button.update(visible=False))),
                        None,
                        [self.submit_btn, self.stop_btn],
                        api_name=False,
                        queue=False,
                    )
            else:
                for event_trigger in event_triggers:
                    event_trigger(
                        async_lambda(lambda: Button.update(visible=True)),
                        None,
                        [self.stop_btn],
                        api_name=False,
                        queue=False,
                    )
                for event_to_cancel in events_to_cancel:
                    event_to_cancel.then(
                        async_lambda(lambda: Button.update(visible=False)),
                        None,
                        [self.stop_btn],
                        api_name=False,
                        queue=False,
                    )
            self.stop_btn.click(
                fn=perform_interrupt,
                inputs=[self.interrupter],
                cancels=events_to_cancel,
                api_name=False,
            )

    def _setup_api(self) -> None:
        api_fn = self._api_stream_fn if self.is_generator else self._api_submit_fn

        self.fake_api_btn.click(
            api_fn,
            [self.textbox, self.chatbot_state] + self.additional_inputs,
            [self.textbox, self.chatbot_state],
            api_name="chat",
        )

    def _clear_and_save_textbox(self, message: str) -> tuple[str, str]:
        return "", message

    async def _display_input(
        self, message: str, history: list[list[str | None]]
    ) -> tuple[list[list[str | None]], list[list[str | None]]]:
        history.append([message, None])
        return history, history

    async def _submit_fn(
        self,
        message: str,
        history_with_input: list[list[str | None]],
        request,  # <- изменили на строку
        *args,
    ) -> tuple[list[list[str | None]], list[list[str | None]]]:
    # ... остальной код без изменений
        history = history_with_input[:-1]
        inputs, _, _ = special_args(
            self.fn, inputs=[message, history, *args], request=request
        )

        if self.is_async:
            response = await self.fn(*inputs)
        else:
            response = await anyio.to_thread.run_sync(
                self.fn, *inputs, limiter=self.limiter
            )

        history.append([message, response])
        return history, history

    async def _stream_fn(
        self,
        message: str,
        history_with_input: list[list[str | None]],
        request,  # <- изменили на строку
        *args,
    ) -> AsyncGenerator:
    # ... остальной код без изменений
        history = history_with_input[:-1]
        inputs, _, _ = special_args(
            self.fn, inputs=[message, history, *args], request=request
        )

        if self.is_async:
            generator = self.fn(*inputs)
        else:
            generator = await anyio.to_thread.run_sync(
                self.fn, *inputs, limiter=self.limiter
            )
            generator = SyncToAsyncIterator(generator, self.limiter)
        try:
            first_response, first_interrupter = await async_iteration(generator)
            update = history + [[message, first_response]]
            yield update, update, first_interrupter
        except StopIteration:
            update = history + [[message, None]]
            yield update, update, first_interrupter
        async for response, interrupter in generator:
            update = history + [[message, response]]
            yield update, update, interrupter

    async def _api_submit_fn(
        self, message: str, history: list[list[str | None]], request, *args  # <- изменили
    ) -> tuple[str, list[list[str | None]]]:
    # ... остальной код без изменений
        inputs, _, _ = special_args(
            self.fn, inputs=[message, history, *args], request=request
        )

        if self.is_async:
            response = await self.fn(*inputs)
        else:
            response = await anyio.to_thread.run_sync(
                self.fn, *inputs, limiter=self.limiter
            )
        history.append([message, response])
        return response, history

    async def _api_stream_fn(
        self, message: str, history: list[list[str | None]], request, *args  # <- изменили
    ) -> AsyncGenerator:
    # ... остальной код без изменений
        inputs, _, _ = special_args(
            self.fn, inputs=[message, history, *args], request=request
        )

        if self.is_async:
            generator = self.fn(*inputs)
        else:
            generator = await anyio.to_thread.run_sync(
                self.fn, *inputs, limiter=self.limiter
            )
            generator = SyncToAsyncIterator(generator, self.limiter)
        try:
            first_response = await async_iteration(generator)
            yield first_response, history + [[message, first_response]]
        except StopIteration:
            yield None, history + [[message, None]]
        async for response in generator:
            yield response, history + [[message, response]]

    async def _delete_prev_fn(
        self,
        message: str,
        history: list[list[str | None]],
    ) -> tuple[
        list[list[str | None]],
        str,
        list[list[str | None]],
    ]:
        while history:
            deleted_a, deleted_b = history[-1]
            history = history[:-1]
            if isinstance(deleted_a, str) and isinstance(deleted_b, str):
                break
        return history, message or "", history