## What X/Y/Z Plot Does
([Читать на русском языке](#что-делает-xyz-plot))

The extension runs a separate generation for each combination of selected values and then arranges the results in a grid. The **X** values go horizontally, **Y** goes vertically, and different **Z** values create separate grids. For example, 3 values for X and 2 for Y result in 6 separate generations and a grid of 3 columns by 2 rows.

Each cell uses the main positive and negative prompts, the model, and other Fooocus settings, changing only the parameters assigned to the axes. X/Y/Z Plot generates **one image per cell**, regardless of the standard **Image Number** field.

## How the Axes Work

For each axis, there is an **X type / Y type / Z type** field and a values field. By default, **Nothing** is selected for all axes. This value yields one base cell and does not change the setting. For standard comparison, assign different parameters to different axes. The same setting can be chosen for more than one axis, but the number of combinations will increase rapidly in that case.

The number of separate generations equals the product of the number of values for X, Y, and Z. For example, 4 X values × 3 Y values × 2 Z values will create 24 images and, if the grid rendering is successful, 2 final grids — one for each Z value. Start with a small test: the more cells, the longer the process.

### How to Enter Values

If the selected setting has a predefined list, a multi-select dropdown and a button to populate the list with available values will appear. This applies, for example, to styles, aspect ratio formats, models, samplers, and schedulers. By default, the list is selected with the mouse.

For parameters without a predefined list, values are entered separated by commas. Numeric axes also understand ranges. Examples:

- `10-30(10)` for an integer parameter means 10, 20, and 30;
- `0.25-0.75[3]` for a fractional parameter sets three points between 0.25 and 0.75 inclusive;
- `10, 20, 30` sets three separate values.

The step is specified in parentheses, and the number of points in the range (including the start and end) is specified in square brackets. For fractional ranges with complex steps, it is more convenient to specify the number of points so that the boundary values are explicitly defined.

Text is also parsed as a comma-separated list. If the text itself contains commas, enclose the corresponding value in quotes, as in a standard CSV.

The **Swap X/Y axes**, **Swap Y/Z axes**, and **Swap X/Z axes** buttons swap the types and values of the two selected axes. This is convenient if you want to swap the layout of rows, columns, and individual grids.

## All Options for X type, Y type, and Z type

Any of the following types can be assigned to X, Y, or Z. The availability of a parameter in the selector does not always mean it is applicable in any configuration: conditions and limitations are noted below.

### Prompts and Styles

- **Nothing** — does not change the task. Used as a single fixed point on the axis.
- **Prompt S/R** — replaces a part of the text in the main positive and negative prompts. The first entered value is the search fragment; all list values become replacement options. For example, for the original prompt `red car on a street`, enter the values `red, blue, green`: the first cell will keep `red`, and the others will replace it with `blue` and `green`. The search text must be present in at least one of the two prompts, otherwise the processor reports that it was not found. Replacement is performed in both prompts if a match exists.
- **Prompt order** — rearranges the specified words or fragments in the positive prompt, creating all possible permutations. Enter fragments that are already in the positive prompt, separated by commas. For 3 fragments, there will be 6 permutations; for 4 — 24. This only changes the order of the named fragments, it does not compose new prompts. Do not include too many fragments: the number of options grows factorially.
- **Styles** — adds the selected style or styles to those already chosen in the main Fooocus interface. The axis does not replace the original set of styles: it supplements it. If the same style is already selected in the main interface, do not duplicate it unnecessarily.

### Main Image Parameters

- **Steps** — the number of generation steps. For comparison, set integer values, for example `20, 30, 40`.
- **Aspect Ratio** — changes the standard format/aspect ratio from the available list. All images for a single final grid must have the same width and height. If different aspect ratios are selected on this axis, the individual images will be saved, but the combined grid may fail to build.
- **Seed** — sets numeric seeds for repeatable comparison. Use a list of specific numbers if you want to compare other settings on fixed seeds.
- **Sharpness** — changes the Fooocus sharpness parameter. Enter integer values, as this axis is configured as an integer type.
- **CFG (Guidance) Scale** — changes the strength of adherence to the text description. Enter one or more numeric values; evaluate the result on your prompts and model.
- **CFG control type** — switches the CFG control type: **Off**, **CFG Mimicking from TSNR**, or **CFG rescale**.
- **CFG Mimicking from TSNR** — changes the value for the mode with the same name. This value makes sense when the corresponding **CFG control type** is selected; with another type, the control may be inactive.
- **CFG rescale** — changes the value for the **CFG rescale** mode. For the comparison to be meaningful, use it together with the CFG rescale type.
- **Clip skip** — changes the integer Clip skip. The effect depends on the model and its text encoder.
- **Softness of ControlNet** — changes the overall softness of the ControlNet influence. Used in tasks with ControlNet-like inputs; this is not a separate setting for a single specific image.

### Model, Refiner, Sampler, Scheduler, and VAE

- **Checkpoint name** — iterates through base models from the list available in the installation. Loading a different model can noticeably increase the run time.
- **Refiner checkpoint** — selects a Refiner from the available models or **None**. If a Refiner is not set, comparing other Refiner parameters usually makes no sense.
- **Refiner switch at** — changes the switching point to the Refiner. Use values allowed in the main interface, and only if a Refiner is selected.
- **Refiner swap method** — selects the Refiner operation method: `joint`, `separate`, or `vae`.
- **Sampler** — iterates through available samplers.
- **Scheduler** — iterates through available schedulers.
- **VAE** — selects **Default (model)** or a VAE file from the list.
- **LoRA 1 name / LoRA 1 weight**, **LoRA 2 name / LoRA 2 weight**, and so on — for each LoRA slot allowed by the specific installation settings, you can change the name or weight. Names are chosen from the list of available LoRAs, including **None**. Weight is entered as a number. If you change the name, keep in mind that this may replace the LoRA specified in the main slot; conversely, the weight axis sets the weight of the LoRA selected in the slot.

### Image Prompt and Other Conditional Input Parameters

The axis list includes separate parameters for **ImagePrompt stop at / ImagePrompt weight**, **PyraCanny stop at / PyraCanny weight**, **CPDS stop at / CPDS weight**, **FaceSwap stop at / FaceSwap weight**, **OpenPose stop at / OpenPose weight**, **Recolor stop at / Recolor weight**, **Scribble stop at / Scribble weight**, and **Manga stop at / Manga weight**. The **stop at** parameter defines up to which part of the process the corresponding condition acts; **weight** changes the strength of its influence.

These axes require that the corresponding input is already configured in the task. If, for example, you select **PyraCanny weight** but do not set a PyraCanny input, the processor will try to modify the first PyraCanny element and may end in an error. When there are multiple elements of the same type in the task, the axis code modifies only the first one. Therefore, first configure the required Image Prompt/ControlNet input in the main interface, then test such an axis with a small number of values.

## Grid Building Settings

- **Draw legend** — enabled by default. Adds X, Y, and Z value labels to the grid. If disabled, the images themselves remain, but the labels on the combined grid are not drawn.
- **Always random seed** — disabled by default. If enabled, a new random seed is assigned to each cell. This overrides the seed that might have been set by the **Seed** axis, so leave the toggle disabled for accurate comparison of specific seeds. With the toggle disabled, all cells without a Seed axis use the seed prepared by the main Fooocus form before launch.
- **Grid margins (px)** — 0 by default; slider range from 0 to 500 with a step of 2. Creates free space between images in the combined grid.
- **White theme of grid** — disabled by default: the grid background is black, and the labels are light. When enabled, the background becomes white, and the labels become dark.

## How to Run a Comparison

1. In the standard Fooocus form, set the positive prompt and, if necessary, the negative prompt. Choose the model, styles, and other parameters that should remain constant.
2. Open the **X/Y/Z Plot** tab.
3. To compare two parameters, select one type on X and another on Y. Leave Z as **Nothing** if a third parameter is not needed.
4. Fill in the values for each axis. For a parameter with a list, select values from the dropdown; for a numeric or text parameter, enter the list/range.
5. Choose the grid settings. Leave **Always random seed** disabled if you are comparing fixed seeds.
6. Before clicking launch, estimate the number of cells as the product of the number of values on the three axes. Shorten the list if the volume turns out to be too large.
7. Launch X/Y/Z Plot. During the process, individual generations and progress notifications are displayed.

Example for 3 CFG values and 2 samplers: assign **CFG (Guidance) Scale** to X with values `3, 5, 7`, **Sampler** to Y, and select two samplers. Leave Z as **Nothing**. 6 images will be created, and the grid will help compare the CFG columns and sampler rows.

## Stop and Skip

The **Skip** button interrupts the current generation, after which the processor continues the queue with the next cell. The **Stop** button terminates the entire X/Y/Z run. Upon Stop, the final combined grid is not built, but the already saved images of completed cells remain.

## Where to Find Results

Individual cell images are saved in the standard Fooocus way in the results directory and displayed in the gallery. The general **Output Format** from Fooocus and the general **Filename Prefix** are used for them. File names include the date, time, and a random suffix.

After completion, the extension separately saves the combined grid: one picture for each Z value. The grid is written to the same general results directory and uses the **Filename Prefix**. Note: the code saves it in the installation's default format, not the **Output Format** chosen for individual images.

The final grid is built only when all images included in it have the same dimensions. Therefore, **Aspect Ratio** can be compared as individual images, but with different sizes, the combined grid is not created in this version.

## Common Problems

- **An error occurs when selecting `Prompt S/R`.** Check that the first value exactly matches a string in the positive or negative prompt.
- **Error on the PyraCanny, ImagePrompt, or other conditional input axis.** First, enable the corresponding input in the main Fooocus form. Such axes change the parameters of an already existing element, rather than creating an input image themselves.
- **Images were created, but there is no combined grid.** Check if you pressed **Stop**, and whether all results have the same dimensions. Different **Aspect Ratio** values usually lead to different sizes.
- **Seed comparison yields non-repeating images.** Disable **Always random seed** and set the numbers via the **Seed** axis.
- **Too many images in the queue.** Reduce the number of values on at least one axis. X/Y/Z Plot creates a generation for every combination.

## Что делает X/Y/Z Plot
([Read in English](#what-xyz-plot-does))

Расширение запускает отдельную генерацию для каждого сочетания выбранных значений, а затем раскладывает результаты в сетку. Значения **X** идут по горизонтали, **Y** — по вертикали, а разные значения **Z** создают отдельные сетки. Например, 3 значения по X и 2 по Y дают 6 отдельных генераций и сетку из 3 столбцов на 2 строки.

Каждая ячейка использует основной положительный и отрицательный промпт, модель и остальные настройки Fooocus, меняя только параметры, назначенные осям. X/Y/Z Plot задаёт **одно изображение на ячейку** независимо от стандартного поля **Image Number**.

## Как устроены оси

Для каждой оси есть поле **X type / Y type / Z type** и поле значений. По умолчанию у всех осей выбран **Nothing**. Такое значение даёт одну базовую ячейку и не меняет настройку. Для обычного сравнения назначьте разные параметры разным осям. Одна и та же настройка может быть выбрана более чем на одной оси, но тогда количество комбинаций быстро увеличивается.

Число отдельных генераций равно произведению количества значений на X, Y и Z. Например, 4 значения X × 3 значения Y × 2 значения Z создадут 24 картинки и, если отрисовка сетки завершится успешно, 2 итоговые сетки — по одной на каждое значение Z. Начните с небольшого теста: чем больше ячеек, тем дольше работа.

### Как вводить значения

Если у выбранной настройки есть готовый список, появится многозначный список выбора и кнопка для заполнения списка доступными значениями. Это относится, например, к стилям, форматам соотношения сторон, моделям, sampler и scheduler. По умолчанию список выбирается мышью.

Для параметров без готового списка значения вводятся через запятую. Числовые оси также понимают диапазоны. Примеры:

- `10-30(10)` для целого параметра означает 10, 20 и 30;
- `0.25-0.75[3]` для дробного параметра задаёт три точки между 0.25 и 0.75 включительно;
- `10, 20, 30` задаёт три отдельных значения.

В круглых скобках указывается шаг, в квадратных — число точек диапазона включая начальную и конечную. Для дробных диапазонов со сложным шагом удобнее указать количество точек, чтобы крайние значения были заданы явно.

Текст тоже разбирается как список через запятую. Если сам текст содержит запятые, заключите соответствующее значение в кавычки, как в обычном CSV.

Кнопки **Swap X/Y axes**, **Swap Y/Z axes** и **Swap X/Z axes** меняют местами типы и значения двух выбранных осей. Это удобно, если вы хотите поменять местами расположение строк, столбцов и отдельных сеток.

## Все варианты `X type`, `Y type` и `Z type`

Любой из следующих типов можно назначить на X, Y или Z. Доступность параметра в селекторе не всегда означает, что он применим в любой конфигурации: ниже отмечены условия и ограничения.

### Промпты и стили

- **Nothing** — не меняет задачу. Используется как одна фиксированная точка на оси.
- **Prompt S/R** — замена части текста в основном положительном и отрицательном промптах. Первое введённое значение — фрагмент для поиска; все значения списка становятся вариантами замены. Например, для исходного промпта `red car on a street` задайте значения `red, blue, green`: первая ячейка оставит `red`, остальные заменят его на `blue` и `green`. Искомый текст должен присутствовать хотя бы в одном из двух промптов, иначе обработка сообщает, что не нашла его. Замена выполняется в обоих промптах, если совпадение есть.
- **Prompt order** — переставляет указанные слова или фрагменты в положительном промпте, создавая все возможные перестановки. Введите фрагменты, которые уже есть в положительном промпте, через запятую. Для 3 фрагментов будет 6 перестановок; для 4 — 24. Это только меняет порядок названных фрагментов, а не составляет новые промпты. Не включайте слишком много фрагментов: количество вариантов растёт факториально.
- **Styles** — добавляет выбранный стиль или стили к уже выбранным в основном интерфейсе Fooocus. Ось не заменяет исходный набор стилей: она дополняет его. Если один и тот же стиль уже выбран в основном интерфейсе, не дублируйте его без необходимости.

### Основные параметры изображения

- **Steps** — число шагов генерации. При сравнении задайте целые значения, например `20, 30, 40`.
- **Aspect Ratio** — меняет стандартный формат/соотношение сторон из доступного списка. Все картинки для одной итоговой сетки должны иметь одинаковую ширину и высоту. Если по этой оси выбраны разные соотношения сторон, сами картинки сохранятся, но общая сетка может не построиться.
- **Seed** — задаёт числовые seed для повторяемого сравнения. Используйте список конкретных чисел, если хотите сравнить другие настройки на фиксированных seed.
- **Sharpness** — меняет параметр резкости Fooocus. Вводите целые значения, поскольку эта ось настроена как целочисленная.
- **CFG (Guidance) Scale** — меняет силу следования текстовому описанию. Введите одно или несколько числовых значений; результат оценивайте на своих промптах и модели.
- **CFG control type** — переключает тип управления CFG: **Off**, **CFG Mimicking from TSNR** или **CFG rescale**.
- **CFG Mimicking from TSNR** — меняет значение для режима с таким же названием. Это значение имеет смысл, когда выбран соответствующий **CFG control type**; при другом типе управление может быть неактивно.
- **CFG rescale** — меняет значение для режима **CFG rescale**. Чтобы сравнение было осмысленным, используйте вместе с типом CFG rescale.
- **Clip skip** — меняет целочисленный Clip skip. Эффект зависит от модели и её текстового энкодера.
- **Softness of ControlNet** — меняет общую мягкость воздействия ControlNet. Используется в задачах с ControlNet-подобными входами; это не отдельная настройка для одной конкретной картинки.

### Модель, Refiner, sampler, scheduler и VAE

- **Checkpoint name** — перебирает базовые модели из списка доступных в установке. Загрузка другой модели может заметно увеличить длительность прогона.
- **Refiner checkpoint** — выбирает Refiner из доступных моделей или **None**. Если Refiner не задан, сравнивать другие параметры Refiner обычно не имеет смысла.
- **Refiner switch at** — меняет точку переключения на Refiner. Используйте значения, допустимые в основном интерфейсе, и только если Refiner выбран.
- **Refiner swap method** — выбирает способ работы Refiner: `joint`, `separate` или `vae`.
- **Sampler** — перебирает доступные sampler-ы.
- **Scheduler** — перебирает доступные scheduler-ы.
- **VAE** — выбирает **Default (model)** либо VAE-файл из списка.
- **LoRA 1 name / LoRA 1 weight**, **LoRA 2 name / LoRA 2 weight** и так далее — для каждого слота LoRA, разрешённого настройками конкретной установки, можно менять имя или вес. Имена выбираются из списка доступных LoRA, включая **None**. Вес вводится числом. Если меняете имя, учитывайте, что это может заменить LoRA, указанную в основном слоте; ось веса, напротив, задаёт вес выбранного в слоте LoRA.

### Параметры Image Prompt и других условных входов

В списке осей есть отдельные параметры **ImagePrompt stop at / ImagePrompt weight**, **PyraCanny stop at / PyraCanny weight**, **CPDS stop at / CPDS weight**, **FaceSwap stop at / FaceSwap weight**, **OpenPose stop at / OpenPose weight**, **Recolor stop at / Recolor weight**, **Scribble stop at / Scribble weight** и **Manga stop at / Manga weight**. Параметр **stop at** задаёт, до какой части процесса действует соответствующее условие; **weight** меняет силу его влияния.

Эти оси требуют, чтобы в задаче уже был настроен соответствующий вход. Если, например, выбрать **PyraCanny weight**, но не задать PyraCanny-вход, обработчик попробует изменить первый PyraCanny-элемент и может завершиться ошибкой. Когда в задаче несколько элементов одного типа, код оси изменяет только первый. Поэтому сначала настройте нужный Image Prompt/ControlNet-вход в основном интерфейсе, затем тестируйте такую ось на небольшом числе значений.

## Настройки построения сетки

- **Draw legend** — включён по умолчанию. Добавляет к сетке подписи значений X, Y и Z. Если отключить, сами картинки остаются, но подписи на общей сетке не рисуются.
- **Always random seed** — выключен по умолчанию. Если включить, для каждой ячейки назначается новый случайный seed. Это переопределяет seed, который мог быть задан осью **Seed**, поэтому оставляйте переключатель выключенным для точного сравнения конкретных seed. При выключенном переключателе все ячейки без оси Seed используют seed, подготовленный основной формой Fooocus перед запуском.
- **Grid margins (px)** — по умолчанию 0; диапазон слайдера от 0 до 500 с шагом 2. Создаёт свободное пространство между картинками в общей сетке.
- **White theme of grid** — выключен по умолчанию: фон сетки чёрный, подписи светлые. При включении фон становится белым, а подписи тёмными.

## Как запустить сравнение

1. В обычной форме Fooocus задайте положительный промпт и при необходимости отрицательный. Выберите модель, стили и прочие параметры, которые должны оставаться постоянными.
2. Откройте вкладку **X/Y/Z Plot**.
3. Для сравнения двух параметров выберите один тип на X и другой на Y. Оставьте Z в **Nothing**, если не нужен третий параметр.
4. Заполните значения каждой оси. Для параметра со списком выберите значения в выпадающем списке; для числового или текстового параметра введите список/диапазон.
5. Выберите настройки сетки. Оставьте **Always random seed** выключенным, если сравниваете фиксированные seed.
6. Перед нажатием на запуск оцените число ячеек как произведение числа значений на трёх осях. Сократите список, если объём оказался слишком большим.
7. Запустите X/Y/Z Plot. В процессе отображаются отдельные генерации и уведомления о ходе работы.

Пример для 3 значений CFG и 2 sampler-ов: назначьте **CFG (Guidance) Scale** на X со значениями `3, 5, 7`, **Sampler** на Y и выберите два sampler-а. Оставьте Z как **Nothing**. Будет создано 6 картинок, а сетка поможет сравнить столбцы CFG и строки sampler-ов.

## Stop и Skip

Кнопка **Skip** прерывает текущую генерацию, после чего обработчик продолжает очередь со следующей ячейкой. Кнопка **Stop** прекращает весь X/Y/Z прогон. При Stop итоговая сводная сетка не строится, но уже сохранённые изображения завершённых ячеек остаются.

## Где искать результаты

Отдельные изображения ячеек сохраняются обычным способом Fooocus в каталоге результатов и отображаются в галерее. Для них используются общий **Output Format** из Fooocus и общий **Filename Prefix**. Имена файлов включают дату, время и случайный суффикс.

После завершения расширение отдельно сохраняет объединённую сетку: по одной картинке на каждое значение Z. Сетка записывается в тот же общий каталог результатов и использует **Filename Prefix**. Обратите внимание: код сохраняет её с форматом по умолчанию установки, а не с выбранным для отдельных изображений **Output Format**.

Итоговая сетка строится только при одинаковых размерах всех входящих в неё изображений. Поэтому **Aspect Ratio** можно сравнить как отдельные картинки, но при разных размерах сводная сетка в этой версии не создаётся.

## Частые проблемы

- **Вылетает ошибка при выборе `Prompt S/R`.** Проверьте, что первое значение точно встречается в положительном или отрицательном промпте.
- **Ошибка на оси PyraCanny, ImagePrompt или другом условном входе.** Сначала включите соответствующий вход в основной форме Fooocus. Такие оси меняют параметры уже существующего элемента, а не создают изображение-вход самостоятельно.
- **Создались картинки, но нет общей сетки.** Проверьте, не нажимали ли **Stop**, и одинаковы ли размеры всех результатов. Разные значения **Aspect Ratio** обычно приводят к разным размерам.
- **Сравнение Seed выдаёт неповторяемые картинки.** Отключите **Always random seed** и задайте числа через ось **Seed**.
- **В очереди слишком много картинок.** Уменьшите количество значений хотя бы на одной оси. X/Y/Z Plot создаёт генерацию для каждой комбинации.

