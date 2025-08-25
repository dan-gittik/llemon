# Llemon

A unified, ergnomoic interface for Generative AI in Python.

- [Installation](#installation)
- [Overview](#overview)
    - [Choosing a Provider](#choosing-a-provider)
    - [Choosing a Model](#choosing-a-model)
    - [Generation](#generation)
        - [Generation Parameters](#generation-parameters)
        - [Streaming](#streaming)
    - [Structured Output](#structured-output)
        - [Classification]
    - [Files](#files)
    - [Tools](#tools)
        - [Builtin Tools](#builtin-tools)
    - [Toolboxes](#toolboxes)
        - [Directory Access](#directory-access)
        - [Database Access](#database-access)
        - [RAG](#rag)
        - [Writing a Custom Toolbox](#writing-a-custom-toolbox)
    - [Prompt Rendering](#prompt-rendering)
        - [Extending Rendering](#extending-rendering)
        - [Builtin Rendering Context](#builtin-rendering-context)
        - [Parallel Rendering](#parallel-rendering)
        - [Toolbox Rendering](#toolbox-rendering)
    - [Conversations](#conversations)
        - [Finalizing Conversations](#finalizing-conversations)
        - [Multi-Bot Conversations](#multi-bot-conversations)
    - [STT](#stt)
    - [TTS](#tts)
    - [Image Generation](#image-generation)
    - [Advanced Topics](#advanced-topics)
        - [Thinking](#thinking)
        - [Embedding](#embedding)
        - [Caching](#caching)
        - [Metrics](#metrics)
- [Local Development](#local-development)
- [License](#license)

![Llemon Logo](llemon.png)

## Installation

From PyPI:

```sh
$ pip install llemon
...
```

From source:

```sh
$ git clone git@github.com:dan-gittik/llemon.git
$ cd llemon/
$ poetry install
...
```

## Overview

Llemon provides identical synchronous and asynchronous APIs. It's async-first,
so `from llemon import ...` exposes the asynchronous version, which we'll use
for the rest of the tutorial; to get equivalent synchronous code, swap it for
`from llemon.sync import ...` and drop the `await`s.

### Choosing a Provider

The first thing we need to do is choose a provider: the same interface works for
OpenAI, Anthropic, Gemini, and various open-source models like Llama or Qwen,
whether we run them locally (e.g. with Ollama) or not (e.g. via DeepInfra or
TogetherAI). Naturally, the various models differ in their capabilities and
costs, so we have to choose which one to use.

Standard models are available as properties:

```pycon
>>> from llemon import OpenAI
>>> OpenAI.gpt5(...)
>>> OpenAI.gpt5_nano(...)

>>> from llemon import Anthropic
>>> Anthropic.opus4(...)
>>> Anthropic.haiku35(...)

>>> from llemon import Gemini
>>> Gemini.pro25(...)
>>> Gemini.lite2(...)

>>> from llemon import DeepInfra
>>> DeepInfra.llama31_70b(...)
>>> DeepInfra.llama31_8b(...)

>>> from llemon import Ollama
>>> Ollama.mistral_7b(...)
```

The only catch is that to access external services, we need to set our API keys.
We can do so by adding them to an `.env` file at the root of our project, with
each API key prefixed with its provider name:

```sh
OPENAI_API_KEY=...

ANTHROPIC_API_KEY=...

GEMINI_API_KEY=...
# Or, if we're using VertexAI:
GEMINI_PROJECT=...
GEMINI_LOCATION=...

DEEPINFRA_API_KEY=...
```

Or configure it programatically with `LLM.configure`:

```pycon
>>> from llemon import LLM
>>> LLM.configure(
...     openai_api_key="...",
...     anthropic_api_key="...",
...     gemini_api_key="...",
...     deepinfra_api_key="...",
... )
```

### Choosing a Model

Like I said, most standard models are available as provider properties. If the
model we need is missing, or we want to choose it based on a dynamic string, we
can do so like this:

```pycon
>>> o1 = OpenAI.model("o1")
```

Bear in mind, though, that each model has a configuration:

```pycon
>>> OpenAI.gpt5.config
LLMModelConfig(
    knowledge_cutoff=datetime.date(2024, 10, 1),
    context_window=400000,
    max_output_tokens=128000,
    supports_streaming=True,
    supports_structured_output=True,
    supports_json=True,
    supports_tools=True,
    accepts_files=['image/jpeg', 'image/png', 'image/gif', 'application/pdf'],
    cost_per_1m_input_tokens=1.25,
    cost_per_1m_output_tokens=10.0,
)
```

Which determines how it work (or doesn't). For example, if structured output is
supported, we can use the model to generate objects based on some schema; if
it's not, we fall back to a JSON implementation; and if JSON is not supported
either, generating an object will raise an error, since there's no way to
guarantee it will adhere to the schema. For predefined models, these
configurations are attached automatically, even if we use strings:

```pycon
>>> gpt5 = OpenAI.model('gpt5')
>>> gpt5.config.supports_structured_output
True
```

For the rest, however, this isn't the case:

```pycon
>>> o3.config.supports_structured_output
None
```

So some features will not work as expected, unless we provide the configuration
(or at least, those parts of it that matter for our scenario) as well:

```pycon
>>> o1 = OpenAI.model("o1", supports_structured_output=True)
```

### Generation

Now that we have a model, let's generate something!

```pycon
>>> response = await OpenAI.gpt5_nano.generate("When was Alan Turing born? Be concise.")
>>> response
<OpenAI/gpt-5-nano: 23 June 1912.>
>>> response.text
'23 June 1912.'
```

If we provide just one string, it's treated as the **user input**; if we provide
two, the first one is treated as the **instructions** (also known as the
**system prompt**):

```pycon
>>> await OpenAI.gpt5_nano.generate("Be concise.", "When was Alan Turing born?")
<OpenAI/gpt-5-nano: 23 June 1912.>
```

Note that using a model directly doesn't retain history: if we were to ask next
"Where?", it'd be a new query, and the model wouldn't know what we're talking
about. To keep the messages history running, we need a **conversation**, which
we'll cover [later](#conversations) – but just to give you a taste:

```pycon
>>> conv = OpenAI.gpt5_nano.conversation("Be concise.")
>>> await conv.generate("When was Alan Turing born?")
<OpenAI/gpt-5-nano: 23 June 1912.>
>>> await conv.generate("Where?")
<OpenAI/gpt-5-nano: Maida Vale, London, England (born at 3 Warrington Crescent).>
```

The conversation object keeps track of all the **interactions** – that is,
request-response pairs – so the entire flow can be inspected:

```pycon
>>> conv
<conversation with OpenAI/gpt-5-nano: 🧑 When was Alan Turing born? | 🤖 23 June 1912. | 🧑 Where? | 🤖 Maida Vale, London, England.>
>>> print(conv)
🧑 When was Alan Turing born?
🤖 23 June 1912.
🧑 Where?
🤖 Maida Vale, London, England.
```

And even indexed or sliced:

```pycon
>>> conv[0]
<conversation with OpenAI/gpt-5-nano: 🧑 When was Alan Turing born? | 🤖 23 June 1912.>
```

This way, we can "undo" (`conv[:x]`) or "forget" (`conv[x:]`) parts of a
conversation and replay it differently ("time travel"):

```pycon
>>> conv = conv[0]
>>> await conv.generate("To whom?")
<OpenAI/gpt-5-nano: Julius Mathison Turing and Sara Turing.>
>>> print(conv)
🧑 When was Alan Turing born?
🤖 23 June 1912.
🧑 To whom?
🤖 Julius Mathison Turing and Sara Turing.
```

#### Generation Parameters

But like I said, more on this later; back to generation for now. In addition to
[files](#files), [tools](#tools), and [prompt rendering](#prompt-rendering),
which we'll address shortly, it accepts all the standard LLM parameters:

- `temperature` (`0-1` or `0-2`, depending on the model; default is usually
  `0.7`): The generation's "originality"; use lower values for more predictable
  responses and higher values for more creative ones.
- `max_tokens` (default is `model.config.max_output_tokens`): The generation's
  maximum output length; use to prevent run-ons or control cost.
- `seed`: the generation's random seed; fix to a specific integer to get
  reproduceable generations (assuming all the other parameters are the same).
- `frequency_penalty` (`-2-2`): Discourages the model from repeating tokens
   proportionally to how often they've appeared; use positive values to reduce
   repetition and negative values to encourage it.
- `presence_penalty` (`-2-2`): Discourages the model from reusing any token that
  has already appeared at least once; use positive values to push the generation
  to introduce new concepts and negative values to keep it focused.
- `top_p` (`0-1`): Sample tokens from the smallest set whose cumulative
  probability ≥ `top_p`; use lower values for more predictable responses and
  higher values for more creative ones (similar to `temperature`; it's
  recommended to use one or the other, but not both).
- `top_k`: Samples only from the top `k` most likely tokens; use lower values to
  keep the response more focused and higher values to make it more diverse.
- `stop`: A list of strings upon which the generation will stop; use for capping
  the output at exact delimiters.
- `prediction`: A projected response that speeds up generation; use when editing
  or rewriting content that is expected to stay largely the same.

It also accept `variants=` (`n=` in OpenAI, `candidate_count=` in Gemini), which
lets us produce multiple responses at once:

```pycon
>>> response = await OpenAI.gpt5_nano.generate("Tell me one sentence about Alan Turing.", variants=3)
>>> response
<OpenAI/gpt-5-nano: Alan Turing poineered computer science and helped crack the Enigma code during WWII.>
```

At first sight, it seems like there's only one response; but that's only because
one variant can be *selected* (i.e. returned by `response.text`) at a given
time, for purposes of keeping a conversation's history coherent. The other
variants are available via `.texts`:

```pycon
>>> response.texts
[
    'Alan Turing pioneered computer science and helped crack the Enigma code during WWII.',
    'At Bletchley Park, Alan Turing chained his tea mug to a radiator to prevent it from being stolen.',
    'Alan Turing was an accomplished long-distance runner, clocking a 2:46 marathon in 1946—close to Olympic standard.',
]
```

And if we prefer a different one, we can `select` it instead (in which case *it*
becomes part of the conversation history):

```pycon
>>> response.select(1)
>>> response
<OpenAI/gpt-5-nano: At Bletchley Park, Alan Turing chained his tea mug to a radiator to prevent it from being stolen.>
```

This is also a good opportunity to show what happens when some model doesn't
support a feature, since Anthropic's Claude doesn't do multiple responses at
once:

```pycon
>>> from llemon import Anthropic
>>> await Anthropic.haiku35.generate("Tell me one sentence about Alan Turing.", variants=3)
llemon.errors.ConfigurationError: Anthropic/claude-3-5-haiku-latest doesn't support multiple responses
```

#### Streaming

Streaming is quite similar to generation, except the response is iterated over
to get the content deltas as they are generated, in real-time:

```pycon
>>> response = await OpenAI.gpt5_nano.stream("When was Alan Turin born?"):
>>> async for delta in response:
...     print(delta.text, end="|")
>>> print()
June| |23|,| |191|2|.|
```

After the stream is exhausted, the accumulated tokens are available as `.text`,
same as before:

```pycon
>>> response.text
'June 23, 1912.'
```

Note that in some models the stream yields one token at a time, like in ChatGPT
above, while in others (like Gemini Flash 2.5) the information is buffered,
yielding parts of (or even entire) sentences at a time:

```pycon
>>> response = Gemini.flash25.stream("When was Alan Turin born?"):
>>> async for delta in response:
...     print(delta.text, end="|")
... print()
June 23, 1912.|
```

**Note**: streaming multiple variants is not supported.

### Structured Output

If we want our generation to adhere to a particular schema, we can define it as
a [Pydantic](https://docs.pydantic.dev/latest/) class and generate an object of
this type:

```pycon
>>> from pydantic import BaseModel
>>> class Person(BaseModel):
...     name: str
...     age: int | None
...     hobbies: list[str]

>>> response = await OpenAI.gpt5_nano.generate_object(
...     Person,
...     "Extract information about the person.",
...     "My name is Alice, and I like reading and hiking.",
... )
>>> response.object
name='Alice' age=None hobbies=['reading', 'hiking']
```

If we pass in a dictionary with a JSON schema, it gets converted to Pydantic
class dynamically:

```pycon
>>> schema = {
...     "title": "Person",
...     "type": "object",
...     "properties": {
...         "name": {
...             "type": "string",
...         },
...         "age": {
...             "type": "integer",
...         },
...         "hobbies": {
...             "type": "array",
...             "items": {
...                 "type": "string",
...             },
...         },
...     },
...     "required": ["name", "hobbies"],
... }

>>> response = await OpenAI.gpt5_nano.generate_object(
...     schema,
...     "Extract information about the person.",
...     "My name is Bob. I'm thirty, and I like cooking.",
... )
>>> response.object
name='Bob' age=30 hobbies=['cooking']
```

Dictionary schemas are more cumbersome to define, and the resulting object types
aren't visible to type checkers, but it can come in handy if we receive the
schema from elsewhere (e.g. via a REST API), in which case converting it to
Pydantic is a bit of a headache. Luckily, Llemon does it for us – and what's
more, it makes sure the schemas adhere to the [subset of features](https://platform.openai.com/docs/guides/structured-outputs#supported-schemas)
supported by LLMs (e.g. dictionary fields are not allowed), raising informative errors if
something about the schema is wrong.

In terms of conversation history, generated objects are stored as
JSON-serialized text – you can see it in the `.text` attribute:

```pycon
>>> response.text
'{"name": "Bob", "age": 25, "hobbies": ["cooking"]}'
```

And just like before, multiple variants can be generated, accessed via
`.objects` (or `.texts`) and `select`ed:

```pycon
>>> response = await OpenAI.gpt5_nano.generate_object(Person, "Generate a person.", variants=3)
>>> response
<OpenAI/gpt-5-nano: name='Jordan Rivera' age=32 hobbies=['baking', 'sci-fi novels']>
>>> response.objects
[
    Person(name='Jordan Rivera', age=32, hobbies=['baking', 'sci-fi novels']),
    Person(name='Alex Morgan', age=25, hobbies=['piano', 'traveling']),
    Person(name='Maya Thompson', age=28, hobbies=['cycling', 'coding']),
]
>>> response.select(2)
>>> response
<OpenAI/gpt-5-nano: name='Maya Thompson' age=28 hobbies=['cycling', 'coding']>
```

#### [WIP] Classification

### Files

Adding files is as simple as passing them in (to any of the generations above)
with the `files=` keyword; those can be string URLs or paths, path objects or
pairs of `(mimetype_or_name, binary_data)`:

```pycon
>>> await OpenAI.gpt5_nano.generate("What's the animal in the picture?", files=["cat.jpg"])
<OpenAI/gpt-5-nano: A tabby cat.>
>>> await OpenAI.gpt5_nano.generate("What's the animal in the picture?", files=["https://picsum.photos/id/237/200/300.jpg"])
<OpenAI/gpt-5-nano: A black dog.>
>>> data = open("cat.jpg", "rb").read()
>>> await OpenAI.gpt5_nano.generate("What's the animal in the picture?", files=[("image/jpeg", data)])
<OpenAI/gpt-5-nano: A tabby cat.>
```

Different models support different files, which are listed in
`model.config.accepts_files`; furthermore, sometimes these files can be inlined
(i.e. sent to the model directly as part of the request), and other times they
have to be uploaded and referenced by ID. All of this happens behind the scenes;
when we use a model directly, the relevant files (e.g. PDFs in case of ChatGPT)
are uploaded before the request and deleted after it. In conversations, where
the model might choose to revisit the files at a later point, the cleanup is
postponed for when `conv.finish()` is called, so it's recommended to use it as a
context manager:

```pycon
>>> async with OpenAI.gpt5_nano.conversation() as conv:
...     response = await conv.generate("What is this story about in one sentence?", files=["story-of-your-life.pdf"]) # file is uploaded
...     print(response)
...     response = await conv.generate("Who is the author?") # file is accessed again
...     print(response)
... # file is deleted
OpenAI/gpt-5-nano: A linguist decodes an alien language that lets her perceive time nonlinearly, enabling her to glimpse her own future and ultimately choose to have a child.
OpenAI/gpt-5-nano: Ted Chiang
```

### Tools

Adding tools is as simple as passing them in (to any of the generations above)
with the `tools=` keyword. The function name is used as the tool name, its
docstring as the tool description, and its parameters are parsed and converted
into an appropriate schema. If the model decides to use a tool, it's called
automatically, and its result (whether it's a return value or an error) is fed
back into the model, so we don't have to do anything other than specify what
capabilities are available:

```pycon
>>> def get_weather(city: str) -> int:
...     "Given a city, return the temperature in Celsius."
...     return {
...         "Paris": 18,
...         "Berlin": 15,
...         "Madrid": 20,
...     }[city]

>>> await OpenAI.gpt5_nano("Where is it hottest, Paris, Berlin or Madrid?", tools=[get_weather])
<OpenAI/gpt-5-nano: Madrid>
```

If we want to see the tools in action, we can enable Llemon's logs:

```pycon
>>> llemon.enable_logs()
>>> response = await OpenAI.gpt5_nano("Where is it hottest, Paris, Berlin or Madrid?", tools=[get_weather])
[08/19/25 20:54:15] DEBUG    🧑 Where is it hottest, Paris, Berlin or Barcelona?
                    DEBUG    🛠️ get_weather(city='Paris')
                    DEBUG    🛠️ get_weather(city='Berlin')
                    DEBUG    🛠️ get_weather(city='Madrid')
                    DEBUG    🛠️ get_weather(city='Paris') returned 18
                    DEBUG    🛠️ get_weather(city='Berlin') returned 15
                    DEBUG    🛠️ get_weather(city='Madrid') returned 20
[08/19/25 20:54:18] DEBUG    🤖 Madrid
```

As you can see, the model chose to call `get_weather` three times, once for each
city, and correctly compared the results. In a conversation, we might prefer to
define such additional capabilities once, and have them available to the model
at any interaction throughout:

```pycon
>>> async with OpenAI.gpt5_nano.conversation(tools=[get_weather]) as conv:
...     response1 = await conv.generate("Where is it hottest, Paris, Berlin or Madrid?")
...     print(response1)
OpenAI/gpt-5-nano: Madrid
```

Then, we can prevent the model from using them on demand:

```pycon
>>> async with OpenAI.gpt5_nano.conversation(tools=[get_weather]) as conv:
...     # … same as before
...     response2 = await conv.generate("And where is it coldest?", use_tool=False)
...     print(response2)
OpenAI/gpt-5-nano: Berlin
```

In this example, the model was able to infer the correct answer despite us
preventing it from invoking `get_weather` again, because tool calls and results
are kept as part of the history, too. In fact, we can inspect them manually in a
given response's `.calls`:

```pycon
>>> response1.calls
[
    <cal 'call_rmwJDaogNIRvMbq7Jcdie1cK': get_weather(city='Paris') -> 18>,
    <cal 'call_8ge90x2cQORINMlBf3Dn41D1': get_weather(city='Berlin') -> 15>,
    <cal 'call_AB25t4wzm3Edfn2gClMc73MK': get_weather(city='Madrid') -> 20>,
]
```

As for the `use_tool=` parameter: if it's `None` (the default), the model can
choose which tools to use at its discretion; if it's `False`, the model is
prohibited from using any; if it's `True`, the model is forced to use at least
one; and if it's a string, the model is forced to use a tool with that
particular name.

Note that in all the cases where a model is allowed to use tools, it might issue
several invocations which are run concurrently: in the asynchronous case, each
is deployed to a separate task, and in the synchronous case each is called from
a separate thread.

#### [WIP] Builtin Tools

### Toolboxes

Llemon comes equipped with several common tools to make life easier; and since
these are customizeable, and thus somewhat more complex, it introduces the
concept of a **toolbox** to define such dynamic collections. First, let's see a
few examples in action, and then cover how to implement one ourselves.

#### Directory Access

Suppose we want to give our model access to the filesystem on our machine; for
exmaple, if we have the following directory structure:

```
files/
    secret.txt # content: "watermelon"
    # … other files
```

And we want the model to access `secret.txt` and tell us its content. Well:

```pycon
>>> from llemon import Directory
>>> response = await OpenAI.gpt5_nano.generate(
...     "What is the secret keyword?",
...     tools=[Directory("files/")],
... )
[08/19/25 21:31:08] DEBUG    🧑 What is the secret keyword?
[08/19/25 21:31:11] DEBUG    🛠️ /home/user/llemon/files.read_files(paths=['secret.txt'])
                    DEBUG    🛠️ /home/user/llemon/files.read_files(paths=['secret.txt']) returned {'secret.txt': 'watermelon'}
[08/19/25 21:31:13] DEBUG    🤖 The secret keyword is: watermelon.
```

By default, directory access is read-only; but if we do:

```pycon
>>> async with OpenAI.gpt5_nano(tools=[Directory("files/", readonly=False)]) as conv:
...     await conv.generate("What is the secret keyword?")
...     await conv.generate("Change it to 'banana'.")
```

We get:

```
[08/19/25 21:34:54] DEBUG    🧑 What is the secret keywoord?
[08/19/25 21:34:58] DEBUG    🛠️ /home/user/llemon/files.read_files(paths=['secret.txt'])
                    DEBUG    🛠️ /home/user/llemon/files.read_files(paths=['secret.txt']) returned {'secret.txt': 'watermelon'}
[08/19/25 21:35:04] DEBUG    🤖 The secret keyword is: watermelon.
                    DEBUG    history
                    DEBUG    GenerateRequest [21:34:54-21:35:04 (9.98s)]
                    DEBUG    🧑 What is the secret keywoord?
                    DEBUG    🛠️  /home/user/llemon/files.read_files(paths=['secret.txt']) -> {'secret.txt': 'watermelon'}
                             🤖 The secret keyword is: watermelon.
                    DEBUG    🧑 Change it to 'banana'.
[08/19/25 21:35:10] DEBUG    🛠️ /home/user/llemon/files.write_files(path='secret.txt', content='banana')
                    DEBUG    🛠️ /home/user/llemon/files.write_files(path='secret.txt', content='banana') returned None
[08/19/25 21:35:14] DEBUG    🛠️ /home/user/llemon/files.read_files(paths=['secret.txt'])
                    DEBUG    🛠️ /home/user/llemon/files.read_files(paths=['secret.txt']) returned {'secret.txt': 'banana'}
[08/19/25 21:35:15] DEBUG    🤖 Secret keyword updated to banana.
```

And check it out – the model actually requested to *read* the file after writing
to it, so as to double-check its contents, which is kind of cute.

#### Database Access

Similarly, we can give our model access to an SQL database. Suppose we have
`db.sqlite3` where:

```sql
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER
);
INSERT INTO users (name, age) VALUES ('Alice', 25);
INSERT INTO users (name, age) VALUES ('Bob', 30);
```

Then:

```pycon
>>> from llemon import Database
>>> async with OpenAI.gpt5_nano(tools=[Database("sqlite:///db.sqlite3")]) as conv:
...     await conv.generate("How old is Alice?")
...     await conv.generate("What about Bob?")
```

Gives us:

```
[08/19/25 22:26:01] DEBUG    🧑 How old is Alice?
[08/19/25 22:26:04] DEBUG    🛠️ sqlite:////home/user/llemon/db.sqlite3.run_sql(sql="SELECT age FROM users WHERE name='Alice';")
                    DEBUG    🛠️ sqlite:////home/user/llemon/db.sqlite3.run_sql(sql="SELECT age FROM users WHERE name='Alice';") returned [{'age': 25}]
[08/19/25 22:26:07] DEBUG    🤖 Alice is 25 years old.
                    DEBUG    history
                    DEBUG    GenerateRequest [18:26:01-18:26:07 (5.76s)]
                    DEBUG    🧑 How old is Alice?
                    DEBUG    🛠️ sqlite:////home/user/llemon/db.sqlite3.run_sql(sql="SELECT age FROM users WHERE name='Alice';") -> [{'age': 25}]
                             🤖 Alice is 25 years old.
                    DEBUG    🧑 What about Bob?
[08/19/25 22:26:10] DEBUG    🛠️ sqlite:////home/user/llemon/db.sqlite3.run_sql(sql="SELECT age FROM users WHERE name='Bob';")
                    DEBUG    🛠️ sqlite:////home/user/llemon/db.sqlite3.run_sql(sql="SELECT age FROM users WHERE name='Bob';") returned [{'age': 30}]
[08/19/25 22:26:12] DEBUG    🤖 Bob is 30 years old.
```

Look at it go, writing its own SQL queries to figure things out! And just like
before, the access is read-only; passing `readonly=False` will allow the model
to create, update and delete records as well.

As an aside, the database toolbox uses asynchronous database drivers, so we need
`aiosqlite` installed for SQLite, `asyncpg` for PostgreSQL and `aiomysql` for
MySQL. If we're using the synchronous version, that'd be the standard `sqlite`
module, `psycopg2` and `pymysql` – but note that we don't have to specify any of
this in the database URL (i.e. don't do `sqlite+aiosqlite://...`).

#### [WIP] RAG

#### [WIP] Writing a Custom Toolbox

### Prompt Rendering

Sometimes, our instructions or messages need to be conditional: maybe they
depend on some context variable, or maybe they should include some bits and not
others depending on our application's state.

Obviously, we can format them ourselves before passing them in – but Llemon
actually comes equipped with template support, courtesy of [Jinja2](https://jinja.palletsprojects.com/en/stable/).
For example, suppose we want to translate English text into some target language
that's part of a student's profile:

```pycon
>>> student.language = "Spanish"
>>> await OpenAI.gpt5_nano.generate(
...     "Translate the following into {{ student.language }}.",
...     "To be, or not to be? That is the question."
...     context={"student": student},
... )
<OpenAI/gpt-5-nano: ¿Ser o no ser? Esa es la cuestión.>
```

Furthermore, suppose that for beginners, we want to add a breakdown of this
translation, and that a student's profile indicates whether they're a beginner
or not:

```pycon
>>> student.is_beginner = True
>>> response = await OpenAI.gpt5_nano.generate(
...     """
...     Translate the following into {{ student.language }}.
...     {% if student.is_beginner %}
...     Add an explanation for beginners, breaking this translation down.
...     {% endif %}
...     """,
...     "To be, or not to be? That is the question.",
...     context={"student": student},
... )
>>> print(response.text)
Translation:
- ¿Ser o no ser? Esa es la cuestión.

Beginner-friendly breakdown:
- ¿ starts a question in Spanish.
- Ser = the infinitive form of the verb “to be.”
- o = or.
- no = not.
- ser = again the verb “to be.”
- ? ends the question.

- Esa = that (demonstrative pronoun).
- es = is (third-person singular of “ser”).
- la = the (feminine definite article).
- cuestión = question/issue; here it means “the issue at stake,” i.e., the main problem or matter.

Whole meaning:
- The line presents a philosophical dilemma using parallel infinitives, and then states that this dilemma is the central issue. In Spanish, “cuestión” captures the idiomatic sense of “that is the question.” An alternative (less common) rendering for nuance is “¿Ser o no ser? Esa es la pregunta.”
```

Also note that prompt indentation is normalized: the first non-empty line's
indent is removed from any line starting with (up to) that many spaces, so if
we're using triple-quoted strings in our code, we don't have to do something
ugly like:

```python
if condition:
    response = OpenAI.gpt5_nano.generate("""
Examples:
  - Example 1.
  - Example 2.
""")
```

But can match our code's indentation without compromising our prompt:

```python
if condition:
    response = OpenAI.gpt5_nano.generate(
        """
        Examples:
          - Example 1.
          - Example 2.
        """
    )
```

Normally, `{{ }}` is used for interpolating expressions, `{% %}` for executing
statements, and `{# #}` for comments. However, if curly braces have a special
meaning in our prompts (for example, we include examples with a lot of
Javascript code), we can change this notation to `<< >>`, `<% %>` and `<# #>`,
`[[ ]]`, `[% %]` and `[# #]` or `(( ))`, `(% %)` and `(# #)` by passing in the
bracket type we want to use to the `render=` keyword:

```pycon
>>> response = await OpenAI.gpt5_nano.generate(
...     "Translate the following into << student.language >>."
...     render="<",
... )
```

#### Extending Rendering

Jinja2 supports most Python operations, but sometimes we might need to extend
it. For example, suppose we want to add a segment to our prompt if a user's
address is in a certain country – the syntax would be:

```
{% if user.address is in_county("US") %}
Mention you're not liable so users don't sue us.
{% endif %}
```

Where `in_county` is own custom function, added to the rendering vocabulary like
so:

```pycon
>>> from llemon import Rendering

... @Rendering.predicate
... def in_country(context, address, country):
...     # check that address is in country…
...     # use the context dictionary to access other template variables if necessary.
```

More generally, for a clause like `{% if value is predicate(*args, **kwargs) %}`
our function's signature should be `predicate(context, value, *args, **kwargs)`.

Similarly, we can define some variables or functions to always be available,
like when we need to inject recurring (and optionally parameterizable) segments:

```
{{ safety_guidelines }}
# or...
{{ safety_guidelines(user.age) }}
```

This is simply a matter of updating `Rendering`'s builtin `namespace`:

```pycon
>>> Rendering.namespace["safety_guidelines"] = "Don't talk about sex, religion or politics."
```

Or using the `function` decorator for functions:

```pycon
>>> @Rendering.function
>>> def safety_guidelines(age):
...     if age < 13:
...         return "Don't talk about sex, religion or politics."
...     return ""
```

#### Toolbox Rendering

Except for custom predicates and namespace functions we define ourselves,
toolboxes can extend formatting, too. `Directory`, for example, lets us inject
its files' contents with:

```pycon
{{ file("path/to/file") }}
```

While `Database` lets us execute an SQL query and embed the resultset in the
instructions – that is, ahead of time, so the model doesn't have to go fishing
for them by itself:

```
{{ sql("SELECT * FROM table") }}
```

#### [WIP] RAG

#### [WIP] Toolbox Rendering Implementation

#### Builtin Rendering Context

A special variable, available during rendering by default, is the `request`
object, representing the current generation request. This can be useful in all
sorts of cases; for example, we can tell the model its knowledge cutoff and
prompt it what to say if asked about events beyond this date:

```
Your knowledge cutoff date is: {{ request.model.config.knowledge_cutoff }}
If asked about events beyond this date, don't improvise; say you don't know.
```

Or use the RAG to inject any context that might be relevant to what the user has
just said:

```
{{ rag(request.user_input) }}
```

Or run some emotional anaysis and respond differently than we normally would
have if the user is distressed:

```
{% if request.user_input is distressed() %}
    # special isntructions
{% else %}
    # regular instructions
{% endif %}
```

#### Parallel Rendering

Rendering supports both synchronous and asynchronous functions – both in case of
predicates (`{% if value is predicate() %}`) and in case of namespace functions
(`{{ function() }}`). However, Jinja2 renders templates *sequentially*, so e.g.
two sleeps of 2 seconds each would take a total of 4 seconds to run:

```pycon
>>> import asyncio, time
>>> from llemon import Rendering
>>> rendering = Rendering("{")

>>> started = time.time()
>>> await rendering.render("""
...     {{ sleep(2) }}
...     {{ async_sleep(2) }}
... """, sleep=time.sleep, async_sleep=asyncio.sleep)
None
None
>>> time.time() - started
4.01
```

Where speed matters, we can parallelize namespace functions by prefixing their
invocation with `!`:

```pycon
>>> started = time.time()
>>> await rendering.render("""
...     {{ !sleep(2) }}
...     {{ !async_sleep(2) }}
... """, sleep=time.sleep, async_sleep=asyncio.sleep)
None
None
>>> time.time() - started
2.01
```

And again, as you can see, this works for both synchronous functions (which are
executed in a threadpool) and asynchronous ones (which are deployed as
concurrent tasks).

This comes in handy with toolbox rendering: if we want to embed the results of
multiple queries, doing this:

```
{{ !sql(query1) }}
{{ !sql(query2) }}
{{ !sql(query3) }}
```

Will take as long as the longest query, instead of their combined time.

Note that this doesn't work for predicates, because here sequentially actually
matters – if we have something like:

```
{% if value1 is predicate1() %}
    {% if value2 is predicate2() %}
    {% endif %}
{% endif %}
```

We'd have to evaluate `predicate1` to know whether we even should (or can)
evaluate `predicate2`; how to parallelize such cases isn't obvious, so it isn't
supported.

So, an alternative pattern is recommended. Suppose we want to run a series of
tests ont he user input (e.g. checking for emotional disress, hostility, or
disengagement) and calibrate the prompt accordingly. These tests might use GenAI
themselves (what's known as "LLM-as-a-judge"), in which case the generation time
can grow significantly; so to parallelize, we can do:

```
{{ !is_distressed(request) }}
{{ !is_hostile(request) }}
{{ !is_disengaged(request) }}

{% if user_is_distressed %}
...
{% elif user_is_hostile %}
...
{% elif user_is_disengaged %}
...
{% else %}
...
{% endif %}
```

Where each of the predicates returns an empty string (i.e. doesn't add a prompt
segment itself), but also modifies the request's context, which is then
available to the rest of the prompt when the parallel evaluation is done and the
rest of is rendered. An implementation of such a function might look like:

```pycon
>>> Rendering.function
... async def is_distressed(request):
...     request.context["user_is_distressed"] = await request.model.classify(
...         f"""
...         Given the following conversation:
...         {request.history[-3:].format(emoji=False)}
...         Would you say the the user is distressed?
...         """,
...         answers=["yes", "no"],
...     ) == "yes"
...     return ""
```

### Conversations

We've already seen how to use conversations to a reasonable extent, but it's
such an important concept in Llemon that it's worth repeating. Calling a model
directly works for simple use-cases, but when we want to have a history of our
interactions – keep previous messages for context, include tool calls and
results so they don't need to be invoked again, and so on – this is the object
we want to be working with.

For convenience, we can create it through a model:

```pycon
>>> conv = OpenAI.gpt5_nano.conversation(...)
```

But we can also create it directly:

```pycon
>>> from llemon import Conversation
>>> conv = Conversation(OpenAI.gpt5_nano, ...)
```

This is useful since conversations can be serialized to a JSON-compatible
dictionary (so the history can be stored in a file or database), then restored
into their original form like so:

```pycon
>>> conv = OpenAI.gpt5_nano.conversation(...)
>>> # interact…
>>> data = conv.dump()
>>> # and later/elsewhere…
>>> conv = Conversation.load(data)
```

Conversations are also a good place to store information that's shared between
multiple interactions: namely, the model's instructions (as well as its context
and rendering, if it's a template) and tools that are always available. Of
course, We can still provide `instructions=`, `context=`, `render=` and
`tools=` per generation, and those override the conversation's "defaults".

If at any point we want to change the conversation's model or any of the above,
we can use the `replace` method to get a copy with the same setup except for
whatever changes we specify:

```pycon
>>> conv2 = conv.replace(instructions="...") # everything else stays the same
```

This is actually what happens when we index or slice the conversation:
`conv[i:j]` is equivalent to `conv.replace(history=conv.history[i:j])`, where
`history` is a list of request-response pairs. We can always see a breakdown of
these if we format the conversation:

```pycon
>>> print(conv.format())
<request 1>
<response 1>
<request 2>
<response 2>
…
```

As well as get it in one line with `one_line=True`, which is what a
conversation's `repr` does, or replace the emojis for `User:`/`Assistant:` texts
with `emoji=False`. Essentially, a conversation is a wrapper around its
`.history` – it even behaves as such. Besides slicing and indexing, `len(conv)`
returns the number of interactions, `for request, response in conv` lets us
iterate over them, and `bool(conv)` tells us if there are any.

Hopefully, this explanation demystifies the concept a little bit: when we call a
generation through a conversation, it does pretty much the same as a standalone
generation does, only it's able to intercept the interaction and append it to
its history. If we want to avoid this for some reason, we can pass `save=False`
to any generation, in which case it will still happen with the full context, but
won't become part of it itself.

#### Finalizing Conversations

The only gotcha is that conversations have lifecycles: for every generation
there might be some necessary preparation (like uploading files) and cleanup
(like deleting them). Since we aim to maintain continuitiy, cleanups are
deferred, and this in turn means we have to be invoke `conv.finish()`
explicitly, after which the conversation is no longer usable:

```pycon
>>> conv.finished
False
>>> await conv.finish()
<conversation: ...>
>>> conv.finished
True
>>> await conv.generate("Actually, one more thing...")
llemon.errors.FinishedError: <conversation: ...> has already finished
```

If we forget to finish a conversation, it will issue a warning when it gets
garbage-collected, pretty much like un-awaited coroutines do:

```pycon
>>> conv.finished
False
>>> del conv
/home/user/llemon/llemon/conversation.py:80: Warning: <conversation: ...> was never finished
```

To make this easier to manage, conversations support the with-statement:

```pycon
>>> async with conv:
...     # interact…
>>> # conversation is finished
```

But note that this isn't always what we want – for example, if we `dump()` the
conversation and `load()` it later, we want to keep any external context (e.g.
files stored at the provider) intact, until we finalize it explicitly. In these
cases, we can suppress the warning with `conv.finish(cleanup=False)`; it's still
encouraged to call this to mark the conversation as over, lest we accidentally
keep using it after we're done. In fact, you might have noticed that it returns
the conversation object itself – this is intended for method chaining, so we can
collect it in one line:

```pycon
>>> data = conv.finish(cleanup=False).dump()
```

And if we ever do want to revive a finished conversation, `conv[:]` ought to do
it. This also brings us to an interesting edge-case: if we replace the
conversation model to a *different provider*, the fact that e.g. eiles were
uploaded in one doesn't mean they will be available in the other – so we need to
(re-)prepare it explicitly, going over (a fresh copy of) its history anew:

```pycon
>>> conv2 = conv.replace(model=Anthropic.haiku35)
>>> await conv2.prepare()
<conversation: ...>
```

And again, with method chaining:

```pycon
>>> conv2 = await conv.replace(model=Anthropic.haiku35).prepare()
```

So, to revive a finished *and* cleaned up conversation:

```pycon
>>> conv2 = await conv[:].prepare()
```

#### Multi-Bot Conversations

To illustrate how useful the conversation object can be, let's run a multi-bot
conversation conversation – something that'd normally be pretty tedious and
confusing to implement.

Suppose we use one LLM to compose a limerick, and another one to criticize it
and give it a grade (`A`, `B` or `C`). We want to iterate between the two until
an `A` grade is reached (and cap it at, say, 10 iterations). Let's use Gemini
and Anthropic, just to mix things up a bit; all the code we need is:

```pycon
>>> class Review(BaseModel):
...     grade: Literal["A", "B", "C"]
...     explanation: str | None

>>> writer = Gemini.lite2("Answer in 5-line limericks, nothing else.")
>>> critic = Anthropic.haiku35("Grade a limerick on its style and wit as A, B or C; if not A, explain what can be improved.")
>>> async with writer, critic:
...     limerick = writer.generate() # empty user input sends "." by default
...     for _ in range(10):
...         review = await critic.generate_object(Review, limerick.text)
...         if review.object.grade == "A":
...             break
...         limerick = await writer.generate(
...             f"""
...             Rewrite your limerick given the following review:
...             {review.object.explanation}
...             """
...         )
...     print(limerick.text)
```

And look at them go:

```
[08/20/25 20:25:12] DEBUG    💡 Answer in 5-line limericks, nothing else.
                    DEBUG    🧑 .
[08/20/25 20:25:13] DEBUG    🤖 A prompt that's quite plain and so bare,
                             Provides naught for a limerick to share.
                             No topic in sight,
                             Makes writing a plight,
                             So I'll stop, 'cause there's truly no there.

                    DEBUG    💡 Grade a limerick on its style and wit as A, B or C; if not A, explain what can be improved.
                    DEBUG    🧑 A prompt that's quite plain and so bare,
                             Provides naught for a limerick to share.
                             No topic in sight,
                             Makes writing a plight,
                             So I'll stop, 'cause there's truly no there.
[08/20/25 20:25:17] DEBUG    🤖 {'grade': 'B', 'explanation': "The limerick has a technically correct AABBA rhyme scheme and follows the traditional meter. However, it lacks
                             the typical playful wit and humor that make limericks memorable. The poem is meta-commentary about the lack of a prompt, which is clever, but,
                             doesn't quite capture the spirited, often cheeky nature of a classic limerick. To improve, it could introduce a more imaginative scenario,
                             include a surprising twist, or use more inventive wordplay that makes the reader chuckle."}

                    DEBUG    💡 Answer in 5-line limericks, nothing else.
                    DEBUG    history
                    DEBUG    GenerateRequest [18:25:12-18:25:13 (0.95s)]
                    DEBUG    🧑 .
                    DEBUG    🤖 A prompt that's quite plain and so bare,
                             Provides naught for a limerick to share.
                             No topic in sight,
                             Makes writing a plight,
                             So I'll stop, 'cause there's truly no there.
                    DEBUG    🧑 Rewrite your limerick given the following review:
                             The limerick has a technically correct AABBA rhyme scheme and follows the traditional meter. However, it lacks the typical playful wit and humor
                             that make limericks memorable. The poem is meta-commentary about the lack of a prompt, which is clever, but doesn't quite capture the spirited,
                             often cheeky nature of a classic limerick. To improve, it could introduce a more imaginative scenario, include a surprising twist, or use more
                             inventive wordplay that makes the reader chuckle.
[08/20/25 20:25:18] DEBUG    🤖 With no prompt, a bard felt quite blue,
                             His muse had all vanished from view.
                             He sighed with a frown,
                             Then thought upside down,
                             And wrote of the *lack* of a cue!

                    DEBUG    💡 Grade a limerick on its style and wit as A, B or C; if not A, explain what can be improved.
                    DEBUG    history
                    DEBUG    GenerateObjectRequest [18:25:13-18:25:17 (4.34s)]
                    DEBUG    🧑 A prompt that's quite plain and so bare,
                             Provides naught for a limerick to share.
                             No topic in sight,
                             Makes writing a plight,
                             So I'll stop, 'cause there's truly no there.
                    DEBUG    🤖 {"grade":"B","explanation":"The limerick has a technically correct AABBA rhyme scheme and follows the traditional meter. However, it lacks the
                             capture the spirited, often cheeky nature of a classic limerick. To improve, it could introduce a more imaginative scenario, include a surprising
                             twist, or use more inventive wordplay that makes the reader chuckle."}
                    DEBUG    🧑 With no prompt, a bard felt quite blue,
                             His muse had all vanished from view.
                             He sighed with a frown,
                             Then thought upside down,
                             And wrote of the *lack* of a cue!
[08/20/25 20:25:22] DEBUG    🤖 {'grade': 'A', 'explanation': None}

With no prompt, a bard felt quite blue,
His muse had all vanished from view.
He sighed with a frown,
Then thought upside down,
And wrote of the *lack* of a cue!
```

**Note**: Handling conversations between more than two bots is currently under
research and development – if you have some clever ideas, get in touch!

### STT

### TTS

### Image Generation

### Advanced Topics

#### Thinking

#### Embeddings

#### Caching

#### Metrics

## Local Development

Install the project with development dependencies:

```sh
$ poetry install --with dev
...
```

The `dev.py` script contains all the development-related tasks, mapped to Poe the Poet commands:

- Linting (with `black`, `isort` and `flake8`):

    ```sh
    $ poe lint [module]*
    ...
    ```

- Type-checking (with `mypy`):

    ```sh
    $ poe type [module]*
    ...
    ```

- Testing (with `pytest`):

    ```sh
    $ poe test [name]*
    ...
    ```

- Coverage (with `pytest-cov`):

    ```sh
    $ poe cov
    ... # browse localhost:8888
    ```

- Clean artefacts generated by these commands:

    ```sh
    $ poe clean
    ```

## License

[MIT](https://opensource.org/license/mit).
