#include "ExecTool.hpp"

#include "ToolNames.hpp"
#include "ToolSchema.hpp"
#include "../llm/Models.hpp"
#include "../utils/JsonUtils.hpp"
#include "../utils/StringUtils.hpp"

#include <simdjson.h>

#ifdef FILO_ENABLE_JAVASCRIPT
#include <quickjs.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <map>
#include <ranges>
#include <thread>
#include <utility>

namespace core::tools {
namespace {

using Clock = std::chrono::steady_clock;

// The default result budget, counted at four bytes a token.
constexpr std::uint64_t kDefaultMaxOutputTokens = 10'000;
constexpr std::size_t kBytesPerToken = 4;
constexpr int kMaxSchemaDepth = 4;

constexpr std::string_view kPragmaPrefix = "// @exec:";

constexpr std::string_view kDescription =
    R"(Run JavaScript that calls tools.
- The input is raw JavaScript source text, not JSON, quoted strings, or markdown code fences. It runs as an async script in a fresh realm, so top-level `await` works. There is no Node, file system, network access, or console.
- Every tool declared below is a method of the global `tools` object, for example `await tools.read({path: "README.md"})`. Tool names become JavaScript identifiers: any character other than a letter, a digit, `_` or `$` becomes `_`. They are not top-level tools: call them only from a script.
- A tool takes an object of its arguments, or a bare string when it has exactly one required string parameter. It resolves to the tool's result: an object when the result is JSON, otherwise a string.
- A call that cannot run (invalid arguments, blocked, or denied by the user) rejects with an Error whose message says why. That is the call's outcome: handle it in the script or adjust, and do not ask for the same permission again.
- Calls issued together, for example with `Promise.all` or `Promise.allSettled`, run in parallel when they do not touch the same files. Batch independent reads and searches in one script.
- The result is the script's status and wall time, then only what the script reported with `text(...)`; the script's own value is not returned. Report what the next decision needs.
- An optional first line `// @exec: {"max_output_tokens": 1000}` sets the result budget, 10000 tokens by default.
- The script runs to completion; promises it did not await are discarded when it ends.
- Global helpers:
- `text(value: string | number | boolean | undefined | null)`: Appends a text item to the result. Non-string values are stringified with `JSON.stringify(...)` when possible.
- `exit()`: Immediately ends the script successfully, like an early return from the top level.
- `store(key: string, value: any)`: Stores a serializable value under a string key for later scripts in the same session.
- `load(key: string)`: Returns the stored value for a key, or `undefined` if it is missing.
- `setTimeout(callback: () => void, delayMs?: number)`: Schedules a callback and returns a timeout id. A pending timeout does not keep the script alive; await a promise to wait for one.
- `clearTimeout(timeoutId?: number)`: Cancels a timeout created by `setTimeout`.
- `ALL_TOOLS`: The callable tools as `{ name, description }` entries.)";

// ── Output ─────────────────────────────────────────────────────────────────

/// The longest prefix no longer than @p bytes that does not split a character.
[[nodiscard]] std::string_view utf8_prefix(std::string_view text, std::size_t bytes) noexcept {
    if (text.size() <= bytes) return text;
    while (bytes > 0 && (static_cast<unsigned char>(text[bytes]) & 0xC0) == 0x80) --bytes;
    return text.substr(0, bytes);
}

/// The longest suffix no longer than @p bytes that starts on a character.
[[nodiscard]] std::string_view utf8_suffix(std::string_view text, std::size_t bytes) noexcept {
    if (text.size() <= bytes) return text;
    std::size_t start = text.size() - bytes;
    while (start < text.size() && (static_cast<unsigned char>(text[start]) & 0xC0) == 0x80) ++start;
    return text.substr(start);
}

/// Keeps the head and the tail, the way a long command output is read.
[[nodiscard]] std::string bounded(std::string text, std::size_t budget) {
    if (text.size() <= budget) return text;
    const auto head = utf8_prefix(text, budget / 2);
    const auto tail = utf8_suffix(text, budget - head.size());
    return std::format("{}\n…[{} bytes truncated]…\n{}",
                       head, text.size() - head.size() - tail.size(), tail);
}

/// `max_output_tokens` from the optional first-line pragma.
[[nodiscard]] std::uint64_t max_output_tokens(std::string_view source) {
    const auto line = core::utils::str::trim_ascii_view(
        source.substr(0, source.find_first_of("\r\n")));
    if (!line.starts_with(kPragmaPrefix)) return kDefaultMaxOutputTokens;
    simdjson::dom::parser parser;
    const simdjson::padded_string json(line.substr(kPragmaPrefix.size()));
    simdjson::dom::element pragma;
    std::uint64_t tokens = 0;
    if (parser.parse(json).get(pragma) != simdjson::SUCCESS
        || pragma["max_output_tokens"].get(tokens) != simdjson::SUCCESS
        || tokens == 0) {
        return kDefaultMaxOutputTokens;
    }
    return tokens;
}

/// The result the model reads: status, wall time, then the script's text items.
[[nodiscard]] std::string render_output(std::string_view status,
                                        Clock::duration wall_time,
                                        std::string items,
                                        std::size_t budget) {
    const double seconds = std::chrono::duration<double>(wall_time).count();
    return std::format("{}\nWall time {:.1f} seconds\nOutput:\n{}",
                       status, seconds, bounded(std::move(items), budget));
}

// ── TypeScript declarations ────────────────────────────────────────────────

[[nodiscard]] std::string render_type(simdjson::dom::element schema, int depth);

[[nodiscard]] std::string render_property_key(std::string_view name) {
    if (script_identifier(name) == name) return std::string(name);
    return "\"" + core::utils::escape_json_string(name) + "\"";
}

[[nodiscard]] bool is_required(simdjson::dom::object schema, std::string_view name) {
    simdjson::dom::array required;
    if (schema["required"].get(required) != simdjson::SUCCESS) return false;
    return std::ranges::any_of(required, [name](simdjson::dom::element entry) {
        std::string_view text;
        return entry.get(text) == simdjson::SUCCESS && text == name;
    });
}

/// `{ a: string; b?: number; }`. The top level of a tool's arguments is
/// written one property per line, each after its description.
[[nodiscard]] std::string render_object(simdjson::dom::object schema, int depth) {
    simdjson::dom::object properties;
    if (schema["properties"].get(properties) != simdjson::SUCCESS) {
        return depth == 0 ? "{}" : "Record<string, unknown>";
    }
    const bool multiline = depth == 0;
    std::string rendered = multiline ? "{\n" : "{ ";
    for (const auto field : properties) {
        if (multiline) {
            std::string_view description;
            if (simdjson::dom::object property;
                field.value.get(property) == simdjson::SUCCESS
                && property["description"].get(description) == simdjson::SUCCESS) {
                for (const auto line : std::views::split(description, '\n')) {
                    const auto text = core::utils::str::trim_ascii_view(
                        std::string_view(line.begin(), line.end()));
                    if (!text.empty()) rendered += std::format("  // {}\n", text);
                }
            }
            rendered += "  ";
        }
        rendered += render_property_key(field.key);
        if (!is_required(schema, field.key)) rendered += '?';
        rendered += ": ";
        rendered += render_type(field.value, depth + 1);
        rendered += multiline ? ";\n" : "; ";
    }
    rendered += '}';
    return rendered;
}

[[nodiscard]] std::string render_union(simdjson::dom::array options, int depth) {
    std::string rendered;
    for (const auto option : options) {
        if (!rendered.empty()) rendered += " | ";
        rendered += render_type(option, depth);
    }
    return rendered.empty() ? "unknown" : rendered;
}

[[nodiscard]] std::string render_named_type(std::string_view type,
                                            simdjson::dom::object schema,
                                            int depth) {
    if (type == "string") return "string";
    if (type == "integer" || type == "number") return "number";
    if (type == "boolean") return "boolean";
    if (type == "null") return "null";
    if (type == "object") return render_object(schema, depth);
    if (type == "array") {
        simdjson::dom::element items;
        if (schema["items"].get(items) != simdjson::SUCCESS) return "Array<unknown>";
        return "Array<" + render_type(items, depth + 1) + ">";
    }
    return "unknown";
}

[[nodiscard]] std::string render_type(simdjson::dom::element schema, int depth) {
    simdjson::dom::object object;
    if (depth > kMaxSchemaDepth || schema.get(object) != simdjson::SUCCESS) return "unknown";

    if (simdjson::dom::array values; object["enum"].get(values) == simdjson::SUCCESS) {
        std::string rendered;
        for (const auto value : values) {
            if (!rendered.empty()) rendered += " | ";
            rendered += simdjson::to_string(value);
        }
        return rendered.empty() ? "never" : rendered;
    }
    if (simdjson::dom::element value; object["const"].get(value) == simdjson::SUCCESS) {
        return std::string(simdjson::to_string(value));
    }
    for (const char* key : {"anyOf", "oneOf"}) {
        if (simdjson::dom::array options; object[key].get(options) == simdjson::SUCCESS) {
            return render_union(options, depth);
        }
    }
    if (std::string_view type; object["type"].get(type) == simdjson::SUCCESS) {
        return render_named_type(type, object, depth);
    }
    if (simdjson::dom::array types; object["type"].get(types) == simdjson::SUCCESS) {
        std::string rendered;
        for (const auto entry : types) {
            std::string_view type;
            if (entry.get(type) != simdjson::SUCCESS) continue;
            if (!rendered.empty()) rendered += " | ";
            rendered += render_named_type(type, object, depth);
        }
        return rendered.empty() ? "unknown" : rendered;
    }
    if (object["properties"].error() == simdjson::SUCCESS) return render_object(object, depth);
    return "unknown";
}

[[nodiscard]] std::string render_arguments(const ToolDefinition& definition) {
    simdjson::dom::parser parser;
    const simdjson::padded_string schema(schema::canonical_input_schema(definition));
    simdjson::dom::element root;
    if (parser.parse(schema).get(root) != simdjson::SUCCESS) return "unknown";
    return render_type(root, 0);
}

/// A heading, the tool's own description, then its TypeScript declaration.
[[nodiscard]] std::string render_tool_section(const ToolDefinition& definition) {
    const std::string identifier = script_identifier(definition.name);
    std::string section = identifier == definition.name
        ? std::format("### `{}`", identifier)
        : std::format("### `{}` (`{}`)", identifier, definition.name);
    const auto description = core::utils::str::trim_ascii_view(definition.description);
    if (!description.empty()) {
        section += '\n';
        section += description;
        section += "\n\n";
    } else {
        section += '\n';
    }
    section += std::format(
        "exec tool declaration:\n```ts\ndeclare const tools: {{ {}(args: {}): Promise<unknown>; }};\n```",
        identifier, render_arguments(definition));
    return section;
}

#ifdef FILO_ENABLE_JAVASCRIPT

// What a script may collect before truncation, so a loop calling text()
// cannot exhaust memory.
constexpr std::size_t kMaxCollectedBytes = 8 * 1024 * 1024;
constexpr std::size_t kMemoryLimitBytes = 64 * 1024 * 1024;
// Scripts run on tool threads, whose native stacks can be as small as 512 KiB.
constexpr std::size_t kMaxStackBytes = 256 * 1024;
constexpr int kMaxToolCalls = 256;
// Uninterrupted JavaScript between two waits is a runaway loop, not work.
constexpr auto kMaxSlice = std::chrono::seconds{30};
// Total time a script may sleep on its own timers.
constexpr auto kMaxTimerWait = std::chrono::seconds{60};
constexpr auto kCancelPoll = std::chrono::milliseconds{50};

// The script-facing helpers, written over the natives the realm installs.
// The natives leave the global object once the helpers hold them.
constexpr std::string_view kPrelude = R"((() => {
  const call = __filo_call, emit = __filo_text, put = __filo_store, get = __filo_load;
  const catalog = __filo_catalog;
  const show = (value) => {
    if (typeof value === "string") return value;
    try {
      const json = JSON.stringify(value);
      return json === undefined ? String(value) : json;
    } catch {
      return String(value);
    }
  };
  const tools = {};
  for (const [identifier, name] of catalog) tools[identifier] = (input) => call(name, input);
  globalThis.tools = Object.freeze(tools);
  globalThis.ALL_TOOLS = Object.freeze(catalog.map(([name, , description]) =>
    Object.freeze({ name, description })));
  globalThis.text = (value) => emit(show(value));
  globalThis.store = (key, value) => put(String(key), JSON.stringify(value));
  globalThis.load = (key) => {
    const json = get(String(key));
    return json === undefined ? undefined : JSON.parse(json);
  };
  globalThis.exit = __filo_exit;
  globalThis.setTimeout = __filo_set_timeout;
  globalThis.clearTimeout = __filo_clear_timeout;
  for (const name of ["__filo_call", "__filo_text", "__filo_store", "__filo_load",
                      "__filo_exit", "__filo_set_timeout", "__filo_clear_timeout",
                      "__filo_catalog"]) {
    delete globalThis[name];
  }
})();)";

// ── QuickJS realm ──────────────────────────────────────────────────────────

/// Owns one reference.
class OwnedValue {
public:
    OwnedValue(JSContext* context, JSValue value) : context_(context), value_(value) {}
    OwnedValue(const OwnedValue&) = delete;
    OwnedValue& operator=(const OwnedValue&) = delete;
    ~OwnedValue() { JS_FreeValue(context_, value_); }

    [[nodiscard]] JSValueConst get() const noexcept { return value_; }
    [[nodiscard]] bool exception() const noexcept { return JS_IsException(value_); }
    /// Hands the reference to a QuickJS call that consumes it.
    [[nodiscard]] JSValue release() noexcept { return std::exchange(value_, JS_UNDEFINED); }

private:
    JSContext* context_;
    JSValue value_;
};

[[nodiscard]] std::string to_text(JSContext* context, JSValueConst value) {
    std::size_t length = 0;
    const char* text = JS_ToCStringLen(context, &length, value);
    if (text == nullptr) {
        JS_FreeValue(context, JS_GetException(context));
        return {};
    }
    std::string copy(text, length);
    JS_FreeCString(context, text);
    return copy;
}

/// `Error: message` and, for an Error, the stack QuickJS recorded.
[[nodiscard]] std::string describe_error(JSContext* context, JSValueConst error) {
    std::string text = to_text(context, error);
    if (JS_IsError(error)) {
        OwnedValue stack(context, JS_GetPropertyStr(context, error, "stack"));
        if (JS_IsString(stack.get())) {
            const std::string frames = to_text(context, stack.get());
            const auto trimmed = core::utils::str::trim_right_view(frames, "\n");
            if (!trimmed.empty()) {
                text += '\n';
                text += trimmed;
            }
        }
    }
    return text.empty() ? std::string("JavaScript error") : text;
}

class ScriptRealm {
public:
    ScriptRealm(ToolScriptHost& host, ToolScriptStore& store)
        : host_(host), store_(store), runtime_(JS_NewRuntime()) {
        if (runtime_ == nullptr) return;
        JS_SetMemoryLimit(runtime_, kMemoryLimitBytes);
        JS_SetMaxStackSize(runtime_, kMaxStackBytes);
        JS_SetInterruptHandler(runtime_, &ScriptRealm::interrupt, this);
        context_ = JS_NewContext(runtime_);
        if (context_ != nullptr) JS_SetContextOpaque(context_, this);
    }

    ScriptRealm(const ScriptRealm&) = delete;
    ScriptRealm& operator=(const ScriptRealm&) = delete;

    ~ScriptRealm() {
        if (context_ != nullptr) {
            for (auto& call : pending_) release(call);
            for (auto& [id, timer] : timers_) JS_FreeValue(context_, timer.callback);
            JS_FreeContext(context_);
        }
        if (runtime_ != nullptr) JS_FreeRuntime(runtime_);
    }

    [[nodiscard]] ToolScriptOutcome run(std::string_view source,
                                        const std::vector<core::llm::Tool>& tools) {
        const auto started = Clock::now();
        const std::size_t budget = static_cast<std::size_t>(std::min<std::uint64_t>(
            max_output_tokens(source) * kBytesPerToken, kMaxCollectedBytes));

        std::optional<std::string> error;
        if (context_ == nullptr) {
            error = "the JavaScript runtime did not start";
        } else if (!install(tools)) {
            error = stop_ == Stop::None
                ? "the script helpers did not start: " + take_exception()
                : failure_of_exception();
        } else {
            // JS_Eval reads one byte past the source: it must be terminated.
            const std::string program(source);
            begin_slice();
            OwnedValue promise(context_, JS_Eval(context_, program.c_str(), program.size(),
                                                 "exec", JS_EVAL_TYPE_GLOBAL | JS_EVAL_FLAG_ASYNC));
            error = promise.exception() ? failure_of_exception() : drive(promise.get());
        }

        std::string items;
        for (const auto& item : items_) {
            if (!items.empty()) items += '\n';
            items += item;
        }
        if (error.has_value()) {
            if (!items.empty()) items += '\n';
            items += "Script error:\n" + *error;
        }
        const std::string_view status = !error.has_value() ? "Script completed"
            : stop_ == Stop::None ? "Script failed"
            : "Script terminated";
        return ToolScriptOutcome{
            .completed = !error.has_value(),
            .output = render_output(status, Clock::now() - started, std::move(items), budget),
        };
    }

private:
    enum class Stop { None, Cancelled, Runaway };

    struct PendingCall {
        ToolScriptCall call;
        JSValue resolve;
        JSValue reject;
    };

    struct Timer {
        JSValue callback;
        Clock::time_point due;
    };

    [[nodiscard]] static ScriptRealm& of(JSContext* context) {
        return *static_cast<ScriptRealm*>(JS_GetContextOpaque(context));
    }

    static int interrupt(JSContext*, void* opaque) {
        auto& self = *static_cast<ScriptRealm*>(opaque);
        if (self.host_.cancelled()) {
            self.stop_ = Stop::Cancelled;
            return 1;
        }
        if (Clock::now() - self.slice_started_ > kMaxSlice) {
            self.stop_ = Stop::Runaway;
            return 1;
        }
        return 0;
    }

    void begin_slice() noexcept { slice_started_ = Clock::now(); }

    void release(PendingCall& call) {
        JS_FreeValue(context_, call.resolve);
        JS_FreeValue(context_, call.reject);
    }

    // ── Natives ──

    static JSValue native_call(JSContext* context, JSValueConst, int argc, JSValueConst* argv) {
        auto& self = of(context);
        if (argc < 1) return JS_ThrowTypeError(context, "a tool call needs a tool name");
        if (++self.calls_ > kMaxToolCalls) {
            return JS_ThrowRangeError(context, "a script may call at most %d tools", kMaxToolCalls);
        }
        ToolScriptCall call{.tool_name = to_text(context, argv[0])};
        const JSValueConst input = argc > 1 ? argv[1] : JS_UNDEFINED;
        if (JS_IsUndefined(input) || JS_IsNull(input)) {
            call.arguments = "{}";
        } else {
            OwnedValue json(context, JS_JSONStringify(context, input, JS_UNDEFINED, JS_UNDEFINED));
            if (json.exception()) return JS_EXCEPTION;
            if (!JS_IsString(json.get())) {
                return JS_ThrowTypeError(context, "a tool input must be a string or an object");
            }
            call.arguments = to_text(context, json.get());
        }
        JSValue resolvers[2];
        const JSValue promise = JS_NewPromiseCapability(context, resolvers);
        if (JS_IsException(promise)) return promise;
        self.pending_.push_back({std::move(call), resolvers[0], resolvers[1]});
        return promise;
    }

    static JSValue native_text(JSContext* context, JSValueConst, int argc, JSValueConst* argv) {
        auto& self = of(context);
        std::string item = argc > 0 ? to_text(context, argv[0]) : std::string("undefined");
        if (self.collected_ < kMaxCollectedBytes) {
            self.collected_ += item.size();
            self.items_.push_back(std::move(item));
        }
        return JS_UNDEFINED;
    }

    static JSValue native_store(JSContext* context, JSValueConst, int argc, JSValueConst* argv) {
        if (argc < 1) return JS_ThrowTypeError(context, "store() needs a key");
        std::optional<std::string> json;
        if (argc > 1 && JS_IsString(argv[1])) json = to_text(context, argv[1]);
        if (!of(context).store_.store(to_text(context, argv[0]), std::move(json))) {
            return JS_ThrowRangeError(context, "store() is full: a session keeps at most %zu bytes",
                                      ToolScriptStore::kMaxBytes);
        }
        return JS_UNDEFINED;
    }

    static JSValue native_load(JSContext* context, JSValueConst, int argc, JSValueConst* argv) {
        if (argc < 1) return JS_UNDEFINED;
        const auto json = of(context).store_.load(to_text(context, argv[0]));
        return json ? JS_NewStringLen(context, json->data(), json->size()) : JS_UNDEFINED;
    }

    static JSValue native_exit(JSContext* context, JSValueConst, int, JSValueConst*) {
        of(context).exited_ = true;
        // Uncatchable, so a try/catch in the script cannot keep it running.
        const JSValue error = JS_NewError(context);
        JS_SetUncatchableError(context, error);
        return JS_Throw(context, error);
    }

    static JSValue native_set_timeout(JSContext* context, JSValueConst, int argc, JSValueConst* argv) {
        auto& self = of(context);
        if (argc < 1 || !JS_IsFunction(context, argv[0])) {
            return JS_ThrowTypeError(context, "setTimeout() needs a callback");
        }
        double delay = 0;
        if (argc > 1 && !JS_IsUndefined(argv[1]) && JS_ToFloat64(context, &delay, argv[1]) < 0) {
            return JS_EXCEPTION;
        }
        if (!std::isfinite(delay) || delay < 0) delay = 0;
        delay = std::min(delay, 1e9);
        const std::int64_t id = ++self.next_timer_id_;
        self.timers_.emplace(id, Timer{
            .callback = JS_DupValue(context, argv[0]),
            .due = Clock::now() + std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double, std::milli>(delay)),
        });
        return JS_NewInt64(context, id);
    }

    static JSValue native_clear_timeout(JSContext* context, JSValueConst, int argc, JSValueConst* argv) {
        auto& self = of(context);
        std::int64_t id = 0;
        if (argc < 1 || JS_IsUndefined(argv[0]) || JS_ToInt64(context, &id, argv[0]) < 0) {
            return JS_UNDEFINED;
        }
        if (const auto timer = self.timers_.find(id); timer != self.timers_.end()) {
            JS_FreeValue(context, timer->second.callback);
            self.timers_.erase(timer);
        }
        return JS_UNDEFINED;
    }

    [[nodiscard]] bool install(const std::vector<core::llm::Tool>& tools) {
        OwnedValue global(context_, JS_GetGlobalObject(context_));
        const auto define = [&](const char* name, JSCFunction* function, int length) {
            return JS_SetPropertyStr(context_, global.get(), name,
                                     JS_NewCFunction(context_, function, name, length)) >= 0;
        };
        if (!define("__filo_call", &native_call, 2) || !define("__filo_text", &native_text, 1)
            || !define("__filo_store", &native_store, 2) || !define("__filo_load", &native_load, 1)
            || !define("__filo_exit", &native_exit, 0)
            || !define("__filo_set_timeout", &native_set_timeout, 2)
            || !define("__filo_clear_timeout", &native_clear_timeout, 1)) {
            return false;
        }

        // [[identifier, name, description], ...]
        std::string catalog = "[";
        for (const auto& tool : tools) {
            if (catalog.size() > 1) catalog += ',';
            catalog += std::format(R"(["{}","{}","{}"])",
                                   core::utils::escape_json_string(script_identifier(tool.function.name)),
                                   core::utils::escape_json_string(tool.function.name),
                                   core::utils::escape_json_string(tool.function.description));
        }
        catalog += ']';
        OwnedValue parsed(context_, JS_ParseJSON(context_, catalog.c_str(), catalog.size(), "catalog"));
        if (parsed.exception()
            || JS_SetPropertyStr(context_, global.get(), "__filo_catalog", parsed.release()) < 0) {
            return false;
        }

        const std::string prelude(kPrelude);
        begin_slice();
        OwnedValue installed(context_, JS_Eval(context_, prelude.c_str(), prelude.size(),
                                               "prelude", JS_EVAL_TYPE_GLOBAL));
        return !installed.exception();
    }

    // ── Event loop ──

    [[nodiscard]] std::string take_exception() {
        OwnedValue exception(context_, JS_GetException(context_));
        return describe_error(context_, exception.get());
    }

    /// Why the script stopped, or nothing when exit() ended it.
    [[nodiscard]] std::optional<std::string> failure_of(JSValueConst error) {
        if (exited_) return std::nullopt;
        switch (stop_) {
        case Stop::Cancelled:
            return "the script was cancelled";
        case Stop::Runaway:
            return std::format("the script ran for more than {} seconds without waiting on a tool",
                               kMaxSlice.count());
        case Stop::None:
            break;
        }
        return describe_error(context_, error);
    }

    [[nodiscard]] std::optional<std::string> failure_of_exception() {
        OwnedValue exception(context_, JS_GetException(context_));
        return failure_of(exception.get());
    }

    [[nodiscard]] std::optional<std::string> drive(JSValueConst promise) {
        for (;;) {
            if (auto failure = drain_jobs()) return failure;
            if (exited_) return std::nullopt;
            switch (JS_PromiseState(context_, promise)) {
            case JS_PROMISE_PENDING:
                break;
            case JS_PROMISE_REJECTED: {
                OwnedValue reason(context_, JS_PromiseResult(context_, promise));
                return failure_of(reason.get());
            }
            case JS_PROMISE_FULFILLED:
            case JS_PROMISE_NOT_A_PROMISE:
                return std::nullopt;
            }
            if (host_.cancelled()) {
                stop_ = Stop::Cancelled;
                return failure_of(JS_UNDEFINED);
            }
            if (!pending_.empty()) {
                if (auto failure = settle_pending_calls()) return failure;
            } else if (!timers_.empty()) {
                if (auto failure = fire_next_timer()) return failure;
            } else {
                return "the script awaited a promise that nothing settles";
            }
        }
    }

    [[nodiscard]] std::optional<std::string> drain_jobs() {
        for (;;) {
            JSContext* job_context = nullptr;
            begin_slice();
            const int ran = JS_ExecutePendingJob(runtime_, &job_context);
            if (ran == 0) return std::nullopt;
            if (ran < 0) return failure_of_exception();
        }
    }

    [[nodiscard]] JSValue reply_value(const ToolScriptReply& reply) {
        if (!reply.ran) {
            const JSValue error = JS_NewError(context_);
            JS_SetPropertyStr(context_, error, "message",
                              JS_NewStringLen(context_, reply.value.data(), reply.value.size()));
            return error;
        }
        const JSValue parsed = JS_ParseJSON(context_, reply.value.c_str(), reply.value.size(), "tool");
        if (!JS_IsException(parsed)) return parsed;
        JS_FreeValue(context_, JS_GetException(context_));
        return JS_NewStringLen(context_, reply.value.data(), reply.value.size());
    }

    [[nodiscard]] std::optional<std::string> settle_pending_calls() {
        auto batch = std::exchange(pending_, {});
        std::vector<ToolScriptCall> calls;
        calls.reserve(batch.size());
        for (const auto& pending : batch) calls.push_back(pending.call);
        auto replies = host_.run(calls);

        std::optional<std::string> failure;
        for (std::size_t i = 0; i < batch.size(); ++i) {
            if (!failure) {
                const ToolScriptReply reply = i < replies.size()
                    ? std::move(replies[i])
                    : ToolScriptReply{.ran = false, .value = "the tool returned no result"};
                JSValue value = reply_value(reply);
                OwnedValue settled(context_, JS_Call(context_, reply.ran ? batch[i].resolve : batch[i].reject,
                                                     JS_UNDEFINED, 1, &value));
                JS_FreeValue(context_, value);
                if (settled.exception()) failure = failure_of_exception();
            }
            release(batch[i]);
        }
        return failure;
    }

    [[nodiscard]] std::optional<std::string> fire_next_timer() {
        const auto next = std::ranges::min_element(timers_, {}, [](const auto& entry) {
            return entry.second.due;
        });
        auto now = Clock::now();
        if (next->second.due > now) {
            if (timer_wait_ + (next->second.due - now) > kMaxTimerWait) {
                return std::format("the script waited on timers for more than {} seconds",
                                   kMaxTimerWait.count());
            }
            while (now < next->second.due) {
                if (host_.cancelled()) {
                    stop_ = Stop::Cancelled;
                    return failure_of(JS_UNDEFINED);
                }
                const auto pause = std::min<Clock::duration>(next->second.due - now, kCancelPoll);
                std::this_thread::sleep_for(pause);
                timer_wait_ += pause;
                now = Clock::now();
            }
        }
        const JSValue callback = next->second.callback;
        timers_.erase(next);
        begin_slice();
        OwnedValue called(context_, JS_Call(context_, callback, JS_UNDEFINED, 0, nullptr));
        JS_FreeValue(context_, callback);
        return called.exception() ? failure_of_exception() : std::nullopt;
    }

    ToolScriptHost& host_;
    ToolScriptStore& store_;
    JSRuntime* runtime_ = nullptr;
    JSContext* context_ = nullptr;
    std::vector<PendingCall> pending_;
    std::map<std::int64_t, Timer> timers_;
    std::int64_t next_timer_id_ = 0;
    Clock::duration timer_wait_{};
    Clock::time_point slice_started_ = Clock::now();
    std::vector<std::string> items_;
    std::size_t collected_ = 0;
    int calls_ = 0;
    bool exited_ = false;
    Stop stop_ = Stop::None;
};

#endif // FILO_ENABLE_JAVASCRIPT

} // namespace

// ── Public API ─────────────────────────────────────────────────────────────

ToolDefinition exec_argument_definition() {
    return ToolDefinition{
        .name = std::string(names::kExec),
        .description = "Run a JavaScript script that calls tools.",
        .parameters = {
            {"source", "string", "JavaScript source.", true},
        },
        // A script may call any offered tool, writes included.
        .annotations = {.destructive_hint = true},
    };
}

std::string arguments_for_source(std::string_view source) {
    return R"({"source":")" + core::utils::escape_json_string(source) + "\"}";
}

std::optional<std::string> source_from_arguments(std::string_view arguments) {
    simdjson::dom::parser parser;
    const simdjson::padded_string padded(arguments);
    simdjson::dom::element document;
    std::string_view source;
    if (parser.parse(padded).get(document) != simdjson::SUCCESS
        || document["source"].get(source) != simdjson::SUCCESS) {
        return std::nullopt;
    }
    return std::string(source);
}

std::string tool_script_description(const std::vector<core::llm::Tool>& tools) {
    std::string description(kDescription);
    for (const auto& tool : tools) {
        description += "\n\n";
        description += render_tool_section(tool.function);
    }
    return description;
}

std::string script_identifier(std::string_view tool_name) {
    std::string identifier;
    identifier.reserve(tool_name.size());
    for (const char ch : tool_name) {
        const bool valid = ch == '_' || ch == '$'
            || (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z')
            || (!identifier.empty() && ch >= '0' && ch <= '9');
        identifier += valid ? ch : '_';
    }
    return identifier.empty() ? std::string("_") : identifier;
}

std::string nested_tool_arguments(const ToolDefinition& definition, std::string arguments) {
    if (!arguments.starts_with('"')) return arguments;
    const ToolParameter* only = nullptr;
    for (const auto& parameter : definition.parameters) {
        if (!parameter.required) continue;
        if (only != nullptr || parameter.type != "string") return arguments;
        only = &parameter;
    }
    if (only == nullptr) return arguments;
    return "{\"" + core::utils::escape_json_string(only->name) + "\":" + arguments + "}";
}

bool ToolScriptStore::store(std::string key, std::optional<std::string> json) {
    std::lock_guard lock(mutex_);
    std::size_t bytes = bytes_;
    if (const auto found = values_.find(key); found != values_.end()) {
        bytes -= found->first.size() + found->second.size();
    }
    if (!json) {
        values_.erase(key);
        bytes_ = bytes;
        return true;
    }
    bytes += key.size() + json->size();
    if (bytes > kMaxBytes) return false;
    values_.insert_or_assign(std::move(key), std::move(*json));
    bytes_ = bytes;
    return true;
}

void ToolScriptStore::clear() {
    std::lock_guard lock(mutex_);
    values_.clear();
    bytes_ = 0;
}

std::optional<std::string> ToolScriptStore::load(const std::string& key) const {
    std::lock_guard lock(mutex_);
    const auto found = values_.find(key);
    if (found == values_.end()) return std::nullopt;
    return found->second;
}

ToolScriptOutcome run_tool_script(std::string_view source,
                                  const std::vector<core::llm::Tool>& tools,
                                  ToolScriptHost& host,
                                  ToolScriptStore& store) {
#ifdef FILO_ENABLE_JAVASCRIPT
    ScriptRealm realm(host, store);
    return realm.run(source, tools);
#else
    (void)tools;
    (void)host;
    (void)store;
    return ToolScriptOutcome{
        .completed = false,
        .output = render_output("Script failed", {},
                                "Script error:\nthis build of Filo has no JavaScript runtime",
                                static_cast<std::size_t>(max_output_tokens(source) * kBytesPerToken)),
    };
#endif
}

} // namespace core::tools
