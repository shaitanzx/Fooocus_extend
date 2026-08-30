from __future__ import annotations
from functools import wraps
import inspect
from typing import AsyncGenerator, Callable

import anyio
from gradio_client import utils as client_utils
from gradio_client.documentation import document, set_documentation_group

import gradio as gr
from gradio.components import (
    Button,
    Chatbot,
    Component,
    Dataset,
    Markdown,
    State,
    Textbox,
    get_component_instance,
)
from gradio.helpers import create_examples as Examples
from gradio.layouts import Accordion, Group, Row
from gradio.themes import ThemeClass as Theme
from gradio.utils import SyncToAsyncIterator, async_iteration

set_documentation_group("chatinterface")


def async_lambda(f: Callable) -> Callable:
    """Turn a function into an async function."""
    @wraps(f)
    async def function_wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return function_wrapper


@document()
class ChatInterface:
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
        clear_btn: str | None | Button = "️  Clear",
        autofocus: bool = True,
    ):
        # Не вызываем super().__init__(), так как мы не наследуем от Blocks

        if post_fn_kwargs is None:
            post_fn_kwargs = {}

        self.post_fn = post_fn
        self.post_fn_kwargs = post_fn_kwargs
        self.pre_fn = pre_fn
        self.pre_fn_kwargs = pre_fn_kwargs

        self.fn = fn
        self.is_async = inspect.iscoroutinefunction(self.fn) or inspect.isasyncgenfunction(self.fn)
        self.is_generator = inspect.isgeneratorfunction(self.fn) or inspect.isasyncgenfunction(self.fn)

        if additional_inputs:
            if not isinstance(additional_inputs, list):
                additional_inputs = [additional_inputs]
            self.additional_inputs = [get_component_instance(i) for i in additional_inputs]
        else:
            self.additional_inputs = []

        if additional_inputs_accordion_name is not None:
            self.additional_inputs_accordion_params = {"label": additional_inputs_accordion_name}
        elif additional_inputs_accordion is None:
            self.additional_inputs_accordion_params = {"label": "Additional Inputs", "open": False}
        elif isinstance(additional_inputs_accordion, str):
            self.additional_inputs_accordion_params = {"label": additional_inputs_accordion}
        elif isinstance(additional_inputs_accordion, Accordion):
            self.additional_inputs_accordion_params = additional_inputs_accordion.recover_kwargs(
                additional_inputs_accordion.get_config()
            )
        else:
            raise ValueError(f"Invalid additional_inputs_accordion type: {type(additional_inputs_accordion)}")

        # Сохраняем параметры для создания компонентов
        self.title = title
        self.description = description
        self.chatbot = chatbot
        self.textbox_param = textbox
        self.submit_btn_param = submit_btn
        self.stop_btn_param = stop_btn
        self.retry_btn_param = retry_btn
        self.undo_btn_param = undo_btn
        self.clear_btn_param = clear_btn
        self.autofocus = autofocus
        self.examples = examples

    def render(self):
        """Создает и возвращает все компоненты интерфейса."""
        components = {}
        
        if self.title:
            components['title'] = Markdown(f"<h1 style='text-align: center; margin-bottom: 1rem'>{self.title}</h1>")
        if self.description:
            components['description'] = Markdown(self.description)

        components['chatbot'] = self.chatbot.render()
        
        with Group():
            with Row():
                if self.textbox_param:
                    self.textbox_param.container = False
                    self.textbox_param.show_label = False
                    textbox_ = self.textbox_param.render()
                    if not isinstance(textbox_, Textbox):
                        raise TypeError(f"Expected a gr.Textbox component, but got {type(textbox_)}")
                    components['textbox'] = textbox_
                else:
                    components['textbox'] = Textbox(
                        container=False, show_label=False, label="Message",
                        placeholder="Type a message...", scale=7, autofocus=self.autofocus,
                    )
                
                if self.submit_btn_param is not None:
                    if isinstance(self.submit_btn_param, Button):
                        components['submit_btn'] = self.submit_btn_param.render()
                    elif isinstance(self.submit_btn_param, str):
                        components['submit_btn'] = Button(self.submit_btn_param, variant="primary", scale=1, min_width=150)
                
                if self.stop_btn_param is not None:
                    if isinstance(self.stop_btn_param, Button):
                        components['stop_btn'] = self.stop_btn_param.render()
                    elif isinstance(self.stop_btn_param, str):
                        components['stop_btn'] = Button(self.stop_btn_param, variant="stop", visible=False, scale=1, min_width=150)
                
                if self.retry_btn_param is not None:
                    if isinstance(self.retry_btn_param, Button):
                        components['retry_btn'] = self.retry_btn_param.render()
                    elif isinstance(self.retry_btn_param, str):
                        components['retry_btn'] = Button(self.retry_btn_param, variant="secondary", scale=1, min_width=150)
                
                if self.undo_btn_param is not None:
                    if isinstance(self.undo_btn_param, Button):
                        components['undo_btn'] = self.undo_btn_param.render()
                    elif isinstance(self.undo_btn_param, str):
                        components['undo_btn'] = Button(self.undo_btn_param, variant="secondary", scale=1, min_width=150)
                
                if self.clear_btn_param is not None:
                    if isinstance(self.clear_btn_param, Button):
                        components['clear_btn'] = self.clear_btn_param.render()
                    elif isinstance(self.clear_btn_param, str):
                        components['clear_btn'] = Button(self.clear_btn_param, variant="secondary", scale=1, min_width=150)

        components['fake_api_btn'] = Button("Fake API", visible=False)
        components['fake_response_textbox'] = Textbox(label="Response", visible=False)

        any_unrendered_inputs = any(not inp.is_rendered for inp in self.additional_inputs)
        if self.additional_inputs and any_unrendered_inputs:
            with Accordion(**self.additional_inputs_accordion_params):
                for input_component in self.additional_inputs:
                    if not input_component.is_rendered:
                        input_component.render()

        components['saved_input'] = State()
        components['chatbot_state'] = State(self.chatbot.value) if self.chatbot.value else State([])
        components['interrupter'] = State(None)

        # Сохраняем компоненты
        self.components = components
        
        return components

    def setup_events(self):
        """Регистрирует все события. Должен вызываться после render()."""
        if not hasattr(self, 'components'):
            raise RuntimeError("Must call render() before setup_events()")
        
        c = self.components
        submit_fn = self._stream_fn if self.is_generator else self._submit_fn

        # Submit event
        submit_event = (
            c['submit_btn'].click(
                self._clear_and_save_textbox, [c['textbox']], [c['textbox'], c['saved_input']],
                api_name=False, queue=False,
            )
            .then(self.pre_fn, **self.pre_fn_kwargs, api_name=False, queue=False)
            .then(self._display_input, [c['saved_input'], c['chatbot_state']], [c['chatbot'], c['chatbot_state']], api_name=False, queue=False)
            .then(submit_fn, [c['saved_input'], c['chatbot_state']] + self.additional_inputs, [c['chatbot'], c['chatbot_state'], c['interrupter']], api_name=False)
            .then(self.post_fn, **self.post_fn_kwargs, api_name=False)
        )

        # Stop button
        if c.get('stop_btn') and self.is_generator:
            def perform_interrupt(ipc):
                if ipc is not None and callable(ipc):
                    ipc()
                return

            c['stop_btn'].click(
                fn=perform_interrupt,
                inputs=[c['interrupter']],
                cancels=[submit_event],
                api_name=False,
                queue=False,
            )

        # Retry button
        if c.get('retry_btn'):
            retry_event = (
                c['retry_btn'].click(
                    self._delete_prev_fn, [c['saved_input'], c['chatbot_state']], 
                    [c['chatbot'], c['saved_input'], c['chatbot_state'], c['textbox']],  
                    api_name=False, queue=False,
                )
                .then(self.pre_fn, **self.pre_fn_kwargs, api_name=False, queue=False)
                .then(self._display_input, [c['saved_input'], c['chatbot_state']], [c['chatbot'], c['chatbot_state']], api_name=False, queue=False)
                .then(submit_fn, [c['saved_input'], c['chatbot_state']] + self.additional_inputs, [c['chatbot'], c['chatbot_state']], api_name=False)
                .then(self.post_fn, **self.post_fn_kwargs, api_name=False)
            )

        # Undo button
        if c.get('undo_btn'):
            c['undo_btn'].click(
                self._delete_prev_fn, [c['saved_input'], c['chatbot_state']], 
                [c['chatbot'], c['saved_input'], c['chatbot_state'], c['textbox']],  
                api_name=False, queue=False,
            ).then(
                self.pre_fn, **self.pre_fn_kwargs, api_name=False, queue=False,
            ).then(
                self.post_fn, **self.post_fn_kwargs, api_name=False, 
            )

        # Clear button
        if c.get('clear_btn'):
            c['clear_btn'].click(
                async_lambda(lambda: ([], [], None, "")), None, 
                [c['chatbot'], c['chatbot_state'], c['saved_input'], c['textbox']], 
                queue=False, api_name=False,
            ).then(
                self.pre_fn, **self.pre_fn_kwargs, api_name=False, queue=False,
            ).then(
                self.post_fn, **self.post_fn_kwargs, api_name=False,
            )

        # Examples
        if self.examples:
            self.examples.click(lambda x: x[0], inputs=[self.examples], outputs=c['textbox'], show_progress=False, queue=False)

        # API
        api_fn = self._api_stream_fn if self.is_generator else self._api_submit_fn
        c['fake_api_btn'].click(
            api_fn, [c['textbox'], c['chatbot_state']] + self.additional_inputs, [c['textbox'], c['chatbot_state']],
            api_name="chat",
        )

    def _clear_and_save_textbox(self, message: str) -> tuple[str, str]:
        return "", message

    async def _display_input(self, message: str, history: list[list[str | None]]) -> tuple[list[list[str | None]], list[list[str | None]]]:
        history.append([message, None])
        return history, history

    async def _submit_fn(self, message: str, history_with_input: list[list[str | None]], *args) -> tuple[list[list[str | None]], list[list[str | None]]]:
        history = history_with_input[:-1]
        inputs = [message, history, *args]
        response = await self.fn(*inputs) if self.is_async else await anyio.to_thread.run_sync(self.fn, *inputs)
        history.append([message, response])
        return history, history

    async def _stream_fn(self, message: str, history_with_input: list[list[str | None]], *args) -> AsyncGenerator:
        history = history_with_input[:-1]
        inputs = [message, history, *args]
        
        if self.is_async:
            generator = self.fn(*inputs)
        else:
            generator = await anyio.to_thread.run_sync(self.fn, *inputs)
            generator = SyncToAsyncIterator(generator, None)
            
        try:
            first_response, first_interrupter = await async_iteration(generator)
            yield history + [[message, first_response]], history + [[message, first_response]], first_interrupter
        except StopIteration:
            yield history + [[message, None]], history + [[message, None]], None
            
        async for response, interrupter in generator:
            yield history + [[message, response]], history + [[message, response]], interrupter

    async def _api_submit_fn(self, message: str, history: list[list[str | None]], *args) -> tuple[str, list[list[str | None]]]:
        inputs = [message, history, *args]
        response = await self.fn(*inputs) if self.is_async else await anyio.to_thread.run_sync(self.fn, *inputs)
        history.append([message, response])
        return response, history

    async def _api_stream_fn(self, message: str, history: list[list[str | None]], *args) -> AsyncGenerator:
        inputs = [message, history, *args]
        if self.is_async:
            generator = self.fn(*inputs)
        else:
            generator = await anyio.to_thread.run_sync(self.fn, *inputs)
            generator = SyncToAsyncIterator(generator, None)
            
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
    ) -> tuple[list[list[str | None]], str, list[list[str | None]], str]:
        if history:
            history = history[:-1]
        return history, message or "", history, message or ""