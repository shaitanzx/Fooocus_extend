## What OMOST Does
([Читать на русском языке](#что-делает-omost))

`OMOST` helps describe an image via chat and then converts the description into a spatial scene layout. It includes a general description of the frame and separate descriptions of objects or regions — for example, "a dragon in the center", "a mountain in the background", "a castle on the right". The OMOST model forms the Canvas structure, and Fooocus uses it during image generation. This is not a regular chat and not an editor where the user draws masks with a mouse.

After the response, two different actions are available:

- **`Render the Image!`** passes the Canvas to the main Fooocus generator. The positive description is built from the Canvas regions; the regular positive prompt line in the main interface does not replace it. The negative prompt and standard Fooocus generation parameters remain part of the task.
- **`Convert to prompt!`** turns the textual description of the Canvas into a single line of a regular prompt and places it in the Prompt field. This button does not generate the image itself: after clicking it, press the standard Fooocus generation button.

OMOST accepts **text messages** here. If you need a composition based on a textual scheme, write it in the chat.

## Where to Open It and What to Prepare

1. In the Fooocus Extend interface, expand **`Extention` → `in generation` → `modules`** and select the **`omost`** tab.
2. In the main interface, check the standard Fooocus settings you usually use in advance: generation model, image resolution, negative prompt, and other parameters. The `Render the Image!` button creates a standard Fooocus task with the current settings.
3. On the first request, the selected OMOST language model is loaded. Before loading, the OMOST code unloads Fooocus models from GPU memory; after the response, the OMOST language model is also unloaded. Therefore, the first message and subsequent rendering may take extra time while the necessary models are loaded back.
4. Write the scene text in the chat field and click **`Submit`**. OMOST models and their training examples are designed to generate structured Canvas descriptions; if the response does not create the desired regions, try rephrasing the request more clearly.

## OMOST Fields and Settings

### Model Selection and Language Model Parameters

The **`LLM model`** dropdown offers six options. The first is selected by default:

- `omost-llama-3-8b-4bits` — default model, 8B in 4-bit format;
- `omost-dolphin-2.9-llama3-8b-4bits`;
- `omost-phi-3-mini-128k-8bits`;
- `omost-llama-3-8b`;
- `omost-dolphin-2.9-llama3-8b`;
- `omost-phi-3-mini-128k`.

These are **OMOST language models** that interpret text and build the Canvas. They are not a Fooocus checkpoint for drawing the final image. The selection applies to the chat operation; the actual rendering is then performed by the current Fooocus generation model. If you switch the model after the first response, `omost` loads the selected model for the next message.

The collapsible **`LLM settings`** block contains the following settings. They control the chat response, not the standard diffusion generation parameters:

- **`Temperature`** — from `0.0` to `2.0`, step `0.01`, default `0.6`. Controls the randomness of OMOST text generation. A value of `0` disables sampling and enables selecting the most probable next token; higher values make the response less predictable. Usually, start with the default value.
- **`Top P`** — from `0.0` to `1.0`, step `0.01`, default `0.9`. Limits the selection of the next token to the most probable group of options. Used when generating the chat response if `Temperature` is not zero. If `Temperature = 0`, the code disables sampling, so `Top P` has practically no effect on this response.
- **`Max New Tokens`** — from `128` to `4096`, default `4096`. Limits the length of the new model response. If the limit is too small for the full Canvas code, the response may cut off and fail to provide a usable scene. For typical use, you can leave it at the maximum; decrease it only if you need to limit response time or size.
- **`Random Seed`** — enabled by default. For each request, OMOST takes a random value; the `Seed Value` field is hidden.
- **`Seed Value`** — an integer, default `12345`, initially hidden. It becomes available when `Random Seed` is disabled; then this value is passed to the OMOST text generator. In the Fooocus code, it is also recorded as `Omost seed` in the task information. This is **not the same as the main Fooocus image seed**: control the final image generation with the regular seed in the main interface.
- **`Use full history`** — disabled by default. When disabled, the next language model request uses only the last completed chat exchange and the new message. When enabled, the entire available correspondence is passed in the request, which helps reference earlier details but increases the context size and may require more resources. The history may still remain visible in the chat window even without this setting.

Changing these parameters does not automatically redraw the already received Canvas. The settings take effect on the next chat request or its rerun.

### Chat Field and Dialog Buttons

The input field is for text messages. The send button is named **`Submit`**. The chat includes **`Quick Prompts`** examples: a request about a battle between warriors and a dragon, and a follow-up suggesting replacing the dragon with a dinosaur. Clicking an example inserts its text into the input field.

- **`New Chat`** — clears the correspondence, the last message, and the current Canvas layout. After clearing, start with a new description.
- **`Edit Last Input`** — returns the last user message to the input field and removes the last exchange from the active history so it can be modified and resent.
- **`Stop`** — stops the streaming generation of the OMOST response while it is in progress.

### Post-Response Buttons: Render or Convert

After a response from which a Canvas could be built, the interface shows the **`Render the Image!`** action and the **`Convert to prompt!`** block. They solve different tasks; choose one path for the current scene.

#### `Render the Image!`

This button passes the Canvas to Fooocus generation. It applies the spatial masks of the regions and their descriptions, rather than just merging them into flat text. The result appears in the standard Fooocus gallery. Resolution, base model, and other parameters are taken from the main generation form.

If the button does not appear after the response, the model likely did not output parsable Canvas code; send a more specific request or edit the last message.

#### `Convert to prompt!` and `Method`

Click **`Convert to prompt!`** to transfer the Canvas descriptions into the regular Prompt field. In the collapsible **`Method`** block, three modes are available. `aggressive` is selected by default.

- **`normal`** — includes general and detailed descriptions, tags, atmosphere, style, and quality metadata if the model provided them. Exact duplicates are removed, but shorter phrases already contained within longer ones are not specifically excluded.
- **`aggressive`** — includes the same details as `normal`, but additionally removes phrases that are substrings of longer phrases. This can make the prompt shorter and reduce repetitions. This is the default value.
- **`short`** — keeps the main scene and region descriptions but excludes detailed descriptions, tags, atmosphere, style, and the quality metadata block. This mode is suitable if you want to get a more concise regular prompt.

In all modes, the result is assembled into a single comma-separated line. After conversion, check the Prompt field and click the standard Fooocus button to generate; `Convert to prompt!` does not create the image itself.

## Step-by-Step Workflow: Scene via Canvas

1. `omost` tab: leave the model `omost-llama-3-8b-4bits` and standard values selected unless you have a specific reason to change them.
2. In the chat, describe the scene and the arrangement of items. For example: "An antique library. Tall bookshelves occupy the background. In the center stands a wooden table with an open book. On the left by the window sits a black cat. Warm evening light."
3. Click `Submit` and wait for the response. If you want to make a correction, send an additional message, for example: "Move the cat to the right side and make the table larger." To reference details from earlier messages, enable `Use full history` before sending the clarification.
4. When the Canvas appears, decide whether you need a spatial render or a regular prompt.
5. For a spatial result, click `Render the Image!`. Rendering is done by the standard Fooocus generator and uses its current settings. The result will appear in the main gallery.
6. For a simple prompt, select the `Method`, click `Convert to prompt!`, check the Prompt field, and start standard Fooocus generation.

## What is Saved and Where to Look for Results

- Chat responses are displayed in the **`Omost chat`** panel. The generated code is used to build the Canvas and its conditions.
- With **`Render the Image!`**, Fooocus launches standard generation and shows the finished images in the standard gallery; the result is saved according to the usual rules and in the usual Fooocus results folder. The number of created images and save parameters are determined by the main Fooocus form.
- With **`Convert to prompt!`**, only text is created in the Prompt field. An image file does not appear until the standard generation is launched.
- The dialog is saved in the state of the current interface until cleared or the session is reset. The **`New Chat`** button removes its content from the active chat and resets the layout for the next request.

## Useful Tips

- If precise object placement is important, use `Render the Image!`: this path preserves separate regional conditions, whereas `Convert to prompt!` reduces them to a sequence of text.
- If the model repeats itself often, try `Method = aggressive` for a regular prompt. If important details are lost, choose `normal`. If you need a concise prompt, choose `short`.
- To correct the scene, write a short clarification and use `Use full history` if the change refers to details mentioned not in the last exchange.
- If the response cuts off before a completed Canvas, increase `Max New Tokens` or simplify the description. In this version, the maximum is already set to 4096.
- If the chat outputs an invalid layout, rephrase the request. The Canvas code is extracted from the response and executed within the application to build the structure.
- Keep memory swapping in mind: when OMOST is working, it unloads Fooocus models from the GPU, and after the response, it unloads the OMOST language model. Repeated operations may include reloading weights into memory (but not from disk) and therefore take more time.

## Troubleshooting

- **No render or conversion buttons after `Submit`.** The buttons appear only if the response could be parsed as a Canvas. Send more specific text, shorten a complex description, or edit the last message.
- **`Render the Image!` reports that the Canvas is empty.** A successful chat response creating a Canvas is required. First, form the scene in the OMOST chat and wait for the response to complete.
- **The render does not match the individual region descriptions.** Check that you clicked exactly `Render the Image!`. The `Convert to prompt!` button forms a regular flat prompt where precise regional masks are not passed. Also, check the main model and Fooocus generation settings.
- **The regular prompt is too long or repetitive.** After the Canvas, select `aggressive`; it removes repeated and nested phrases. If the short prompt lacks details, switch to `normal`.
- **The scene cuts off or the Canvas is not parsed.** Increase the `Max New Tokens` limit if it was decreased, and ask the model to describe fewer objects at once. The code cannot process an incomplete or syntactically incorrect response as a normal Canvas. To retry, send the request again or change it via `Edit Last Input`.
- **Loading the first model takes a long time or Fooocus generation after the chat is slow.** The language model is loaded into the cache, and the code frees up GPU memory by unloading Fooocus models before the chat works. Let the first stage complete; after the chat, Fooocus may load its models back.

## Что делает OMOST
([Read in English](#what-omost-does))

`OMOST` помогает описать изображение через чат, а затем преобразовать описание в пространственную схему сцены. В ней есть общее описание кадра и отдельные описания объектов или областей — например, «дракон в центре», «гора на заднем плане», «замок справа». Модель OMOST формирует структуру Canvas, а Fooocus использует её при генерации изображения. Это не обычный чат и не редактор, где пользователь рисует маски мышью.

После ответа доступны два разных действия:

- **`Render the Image!`** передаёт Canvas в основной генератор Fooocus. Положительное описание строится из областей Canvas; обычная положительная строка промпта в главном интерфейсе не заменяет его. Отрицательный промпт и стандартные параметры генерации Fooocus остаются частью задания.
- **`Convert to prompt!`** превращает текстовое описание Canvas в одну строку обычного промпта и помещает её в поле Prompt. Само изображение эта кнопка не генерирует: после неё нажмите стандартную кнопку генерации Fooocus.

OMOST принимает здесь **текстовые сообщения**. Если нужна композиция по текстовой схеме, напишите её в чате.

## Где открыть и что подготовить

1. В интерфейсе Fooocus Extend раскройте **`Extention` → `in generation` → `modules`** и выберите вкладку **`omost`**.
2. В главном интерфейсе заранее проверьте обычные настройки Fooocus, которыми вы обычно пользуетесь: модель генерации, размер изображения, отрицательный промпт и другие параметры. Кнопка `Render the Image!` создаёт обычное задание Fooocus с текущими настройками.
3. При первом обращении загружается выбранная языковая модель OMOST. До загрузки OMOST код выгружает модели Fooocus из памяти GPU; после ответа языковая модель OMOST также выгружается. Поэтому первое сообщение и последующий рендер могут занять дополнительное время, пока нужные модели загружаются обратно.
4. Напишите текст сцены в поле чата и нажмите **`Submit`**. Модели OMOST и их обучающие примеры рассчитаны на генерацию структурированных Canvas-описаний; если ответ не создаёт нужные области, попробуйте переформулировать запрос яснее.

## Поля и настройки OMOST

### Выбор модели и параметры языковой модели

В выпадающем списке **`LLM model`** доступны шесть вариантов. По умолчанию выбран первый:

- `omost-llama-3-8b-4bits` — модель по умолчанию, 8B в 4-битном варианте;
- `omost-dolphin-2.9-llama3-8b-4bits`;
- `omost-phi-3-mini-128k-8bits`;
- `omost-llama-3-8b`;
- `omost-dolphin-2.9-llama3-8b`;
- `omost-phi-3-mini-128k`.

Это **языковые модели OMOST**, которые интерпретируют текст и составляют Canvas. Они не являются checkpoint’ом Fooocus для рисования итоговой картинки. Выбор применяется к работе чата; собственно рендер затем выполняется текущей моделью генерации Fooocus. Если переключить модель после первого ответа, `omost` загружает выбранную модель для следующего сообщения.

В раскрываемом блоке **`LLM settings`** находятся следующие настройки. Они управляют ответом чата, а не обычными параметрами диффузионной генерации:

- **`Temperature`** — от `0.0` до `2.0`, шаг `0.01`, по умолчанию `0.6`. Управляет случайностью генерации текста OMOST. Значение `0` отключает sampling и включает выбор наиболее вероятного следующего токена; более высокие значения делают ответ менее предсказуемым. Обычно начните со значения по умолчанию.
- **`Top P`** — от `0.0` до `1.0`, шаг `0.01`, по умолчанию `0.9`. Ограничивает выбор следующего токена наиболее вероятной группой вариантов. Используется при генерации ответа чата, когда `Temperature` не равна нулю. При `Temperature = 0` код отключает sampling, поэтому `Top P` практически не влияет на этот ответ.
- **`Max New Tokens`** — от `128` до `4096`, по умолчанию `4096`. Ограничивает длину нового ответа модели. Если лимит слишком мал для полного кода Canvas, ответ может оборваться и не дать пригодной сцены. При типичном использовании можно оставить максимум; уменьшайте его только если нужно ограничить время или объём ответа.
- **`Random Seed`** — по умолчанию включён. Для каждого запроса OMOST берёт случайное значение; поле `Seed Value` скрыто.
- **`Seed Value`** — целое число, начальное значение `12345`, изначально скрыто. Оно становится доступным, когда выключен `Random Seed`; тогда это значение передаётся генератору текста OMOST. В коде Fooocus оно также записывается как `Omost seed` в информации о задании. Это **не то же самое, что основной seed изображения Fooocus**: итоговую генерацию изображения контролируйте обычным seed в главном интерфейсе.
- **`Use full history`** — по умолчанию выключен. Когда выключен, следующий запрос языковой модели использует только последний завершённый обмен чата и новое сообщение. Когда включён, в запрос передаётся вся доступная переписка, что помогает ссылаться на более ранние детали, но увеличивает объём контекста и может требовать больше ресурсов. История всё равно может оставаться видна в окне чата и без этой настройки.

Изменение этих параметров не перерисовывает уже полученный Canvas автоматически. Настройки вступают в силу при следующем запросе в чат или его повторном запуске.

### Поле чата и кнопки диалога

Поле ввода — текстовое сообщение. Кнопка отправки называется **`Submit`**. В чате есть примеры **`Quick Prompts`**: запрос про битву воинов с драконом и продолжение, предлагающее заменить дракона динозавром. Нажатие на пример подставляет его текст в поле ввода.

- **`New Chat`** — очищает переписку, последнее сообщение и текущую схему Canvas. После очистки начните с нового описания.
- **`Edit Last Input`** — возвращает последнее пользовательское сообщение в поле ввода и удаляет последний обмен из активной истории, чтобы его можно было изменить и отправить заново.
- **`Stop`** — останавливает потоковую генерацию ответа OMOST, пока она выполняется.

### Кнопки после ответа: рендер или конвертация

После ответа, из которого удалось построить Canvas, интерфейс показывает действие **`Render the Image!`** и блок **`Convert to prompt!`**. Они решают разные задачи; выберите один путь для текущей сцены.

#### `Render the Image!`

Эта кнопка передаёт Canvas на генерацию Fooocus. Она применяет пространственные маски регионов и их описания, а не только сводит их в плоский текст. Итог появляется в стандартной галерее Fooocus. Размер, основная модель и другие параметры берутся из главной формы генерации.

Если кнопка не появилась после ответа, вероятно, модель не выдала разобранный код Canvas; отправьте более конкретный запрос или отредактируйте последнее сообщение.

#### `Convert to prompt!` и `Method`

Нажмите **`Convert to prompt!`**, чтобы перенести описания Canvas в обычное поле Prompt. В выпадающем блоке **`Method`** доступны три режима. По умолчанию выбран `aggressive`.

- **`normal`** — включает общие и подробные описания, теги, атмосферу, стиль и метаданные качества, если модель их выдала. Удаляются точные повторы, но более короткие фразы, уже содержащиеся внутри длинных, специально не исключаются.
- **`aggressive`** — включает те же подробности, что `normal`, но дополнительно убирает фразы, являющиеся подстроками более длинных фраз. Это может сделать промпт короче и уменьшить повторения. Значение по умолчанию.
- **`short`** — оставляет основные описания сцены и регионов, но исключает подробные описания, теги, атмосферу, стиль и блок метаданных качества. Режим подходит, если хочется получить более краткий обычный промпт.

Во всех режимах результат собирается в одну строку через запятые. После конвертации проверьте поле Prompt и нажмите стандартную кнопку Fooocus для генерации; `Convert to prompt!` само изображение не создаёт.

## Пошаговая работа: сцена через Canvas

1. Вкладка `omost`: оставьте выбранной модель `omost-llama-3-8b-4bits` и стандартные значения, если у вас нет особой причины их менять.
2. В чате опишите сцену и расположение предметов. Например: «Старинная библиотека. Высокие книжные шкафы занимают фон. В центре стоит деревянный стол с раскрытой книгой. Слева у окна сидит чёрный кот. Тёплый вечерний свет».
3. Нажмите `Submit` и дождитесь ответа. Если хотите внести поправку, отправьте дополнительное сообщение, например: «Перемести кота на правую сторону и сделай стол крупнее». Для ссылок на детали из более ранних сообщений включите `Use full history` до отправки уточнения.
4. Когда Canvas появился, решите, нужен ли пространственный рендер или обычный промпт.
5. Для пространственного результата нажмите `Render the Image!`. Рендер идёт стандартным генератором Fooocus и использует его текущие настройки. Результат появится в главной галерее.
6. Для простого промпта выберите `Method`, нажмите `Convert to prompt!`, проверьте поле Prompt и запустите обычную генерацию Fooocus.

## Что сохраняется и где смотреть результат

- Ответы чата показываются в панели **`Omost chat`**. Сгенерированный код используется для построения Canvas и его условий.
- При **`Render the Image!`** Fooocus запускает стандартную генерацию и показывает готовые изображения в стандартной галерее; результат сохраняется по обычным правилам и в обычную папку результатов Fooocus. Число создаваемых картинок и параметры сохранения определяются основной формой Fooocus.
- При **`Convert to prompt!`** создаётся только текст в поле Prompt. Файл изображения не появляется до стандартного запуска генерации.
- Диалог сохраняется в состоянии текущего интерфейса до очистки или сброса сессии. Кнопка **`New Chat`** удаляет его содержимое из активного чата и сбрасывает схему для следующего запроса.

## Полезные советы

- Если важно точное расположение объектов, используйте `Render the Image!`: этот путь сохраняет раздельные региональные условия, тогда как `Convert to prompt!` сводит их к последовательности текста.
- Если модель часто повторяется, попробуйте `Method = aggressive` для обычного промпта. Если теряются важные пояснения, выберите `normal`. Если нужен лаконичный prompt, выберите `short`.
- Для исправления сцены напишите короткое уточнение и используйте `Use full history`, если изменение ссылается на детали, упомянутые не в последнем обмене.
- Если ответ обрывается до завершённого Canvas, увеличьте `Max New Tokens` или упростите описание. В этой версии максимум уже равен 4096.
- Если чат выдаёт невалидную схему сформулируйте запрос заново. Код Canvas извлекается из ответа и выполняется внутри приложения для построения структуры.
- Учтите переключение памяти: при работе OMOST выгружает модели Fooocus из GPU, а после ответа выгружает языковую модель OMOST. Повторные операции могут включать повторную загрузку весов в память (но не на диск) и потому занимать больше времени.

## Устранение неполадок

- **После `Submit` нет кнопок рендера и конвертации.** Кнопки показываются, если ответ удалось разобрать как Canvas. Отправьте более конкретный текст, сократите сложное описание или отредактируйте последнее сообщение.
- **`Render the Image!` сообщает, что Canvas пуст.** Нужен успешный ответ чата, создающий Canvas. Сначала сформируйте сцену в OMOST-чате и дождитесь завершения ответа.
- **Рендер не соответствует отдельным описаниям областей.** Проверьте, что нажали именно `Render the Image!`. Кнопка `Convert to prompt!` формирует обычный плоский промпт, где точные региональные маски не передаются. Также проверьте основную модель и настройки генерации Fooocus.
- **Обычный промпт слишком длинный или повторяется.** После Canvas выберите `aggressive`; он удаляет повторные и вложенные фразы. Если в кратком промпте не хватает деталей, переключитесь на `normal`.
- **Сцена обрывается или Canvas не разбирается.** Увеличьте лимит `Max New Tokens`, если он был уменьшен, и попросите модель описать меньше объектов за один раз. Код не может обработать неполный или синтаксически неверный ответ как нормальный Canvas. Для повтора отправьте запрос ещё раз или измените его через `Edit Last Input`.
- **Загрузка первой модели занимает долгое время или генерация Fooocus после чата медленная.** Языковая модель загружается в кэш, а код освобождает память GPU, выгружая модели Fooocus до работы чата. Дайте первому этапу завершиться; после чата Fooocus может загружать свои модели обратно.
