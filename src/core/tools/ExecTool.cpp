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
#include <span>
#include <thread>
#include <tuple>
#include <utility>

namespace core::tools {
namespace {

using Clock = std::chrono::steady_clock;

// The default result budget, counted at four bytes a token.
constexpr std::uint64_t kDefaultMaxOutputTokens = 10'000;
constexpr std::size_t kBytesPerToken = 4;
constexpr int kMaxSchemaDepth = 4;

constexpr std::string_view kPragmaPrefix = "// @exec:";

// codex-rs code-mode-protocol/src/description.rs MCP_TYPESCRIPT_PREAMBLE:
// the shared result shape an MCP tool returns, so a script can read
// result.content without guessing.
constexpr std::string_view kMcpTypescriptPreamble = R"(type Role = "user" | "assistant";
type MetaObject = Record<string, unknown>;
type Annotations = {
  audience?: Role[];
  priority?: number;
  lastModified?: string;
};
type Icon = {
  src: string;
  mimeType?: string;
  sizes?: string[];
  theme?: "light" | "dark";
};
type TextResourceContents = {
  uri: string;
  mimeType?: string;
  _meta?: MetaObject;
  text: string;
};
type BlobResourceContents = {
  uri: string;
  mimeType?: string;
  _meta?: MetaObject;
  blob: string;
};
type TextContent = {
  type: "text";
  text: string;
  annotations?: Annotations;
  _meta?: MetaObject;
};
type ImageContent = {
  type: "image";
  data: string;
  mimeType: string;
  annotations?: Annotations;
  _meta?: MetaObject;
};
type AudioContent = {
  type: "audio";
  data: string;
  mimeType: string;
  annotations?: Annotations;
  _meta?: MetaObject;
};
type ResourceLink = {
  icons?: Icon[];
  name: string;
  title?: string;
  uri: string;
  description?: string;
  mimeType?: string;
  annotations?: Annotations;
  size?: number;
  _meta?: MetaObject;
  type: "resource_link";
};
type EmbeddedResource = {
  type: "resource";
  resource: TextResourceContents | BlobResourceContents;
  annotations?: Annotations;
  _meta?: MetaObject;
};
type ContentBlock =
  | TextContent
  | ImageContent
  | AudioContent
  | ResourceLink
  | EmbeddedResource;
type CallToolResult<TStructured = { [key: string]: unknown }> = {
  _meta?: MetaObject;
  content: ContentBlock[];
  isError?: boolean;
  structuredContent?: TStructured;
  [key: string]: unknown;
};)";

struct ExecPragma {
    std::uint64_t max_output_tokens = kDefaultMaxOutputTokens;
    std::optional<std::uint64_t> yield_time_ms;
};

/// `max_output_tokens` and `yield_time_ms` from the optional first-line pragma.
[[nodiscard]] ExecPragma parse_exec_pragma(std::string_view source) {
    ExecPragma pragma;
    const auto line = core::utils::str::trim_ascii_view(
        source.substr(0, source.find_first_of("\r\n")));
    if (!line.starts_with(kPragmaPrefix)) return pragma;
    simdjson::dom::parser parser;
    const simdjson::padded_string json(line.substr(kPragmaPrefix.size()));
    simdjson::dom::element document;
    if (parser.parse(json).get(document) != simdjson::SUCCESS) return pragma;
    std::uint64_t tokens = 0;
    if (document["max_output_tokens"].get(tokens) == simdjson::SUCCESS && tokens > 0) {
        pragma.max_output_tokens = tokens;
    }
    std::uint64_t yield_ms = 0;
    if (document["yield_time_ms"].get(yield_ms) == simdjson::SUCCESS) {
        pragma.yield_time_ms = yield_ms;
    }
    return pragma;
}

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

/// The result the model reads: status, wall time, the script's text items
/// within @p budget, then what it sent with notify(), which no budget cuts.
[[nodiscard]] std::string render_output(std::string_view status,
                                        Clock::duration wall_time,
                                        std::string items,
                                        std::size_t budget,
                                        std::span<const std::string> notes = {}) {
    const double seconds = std::chrono::duration<double>(wall_time).count();
    std::string output = std::format("{}\nWall time {:.1f} seconds\nOutput:\n{}",
                                     status, seconds, bounded(std::move(items), budget));
    for (const auto& note : notes) {
        output += "\nNotify:\n";
        output += note;
    }
    return output;
}

/// The MIME type of a base64 image data: URL, or nothing for anything else.
[[nodiscard]] std::optional<std::string_view> image_data_url_mime(std::string_view url) {
    constexpr std::string_view kPrefix = "data:image/";
    constexpr std::string_view kBase64 = ";base64,";
    if (!url.starts_with(kPrefix)) return std::nullopt;
    const auto marker = url.find(kBase64);
    if (marker == std::string_view::npos || marker + kBase64.size() >= url.size()) return std::nullopt;
    return url.substr(5, marker - 5);
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

/// What the call resolves to, from the tool's output schema (codex-rs
/// render_code_mode_sample_for_definition), or `unknown` without one.
[[nodiscard]] std::string render_result(const ToolDefinition& definition) {
    if (definition.output_schema.empty()) return "unknown";
    simdjson::dom::parser parser;
    const simdjson::padded_string schema(definition.output_schema);
    simdjson::dom::element root;
    if (parser.parse(schema).get(root) != simdjson::SUCCESS) return "unknown";
    return render_type(root, 1);
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
        "exec tool declaration:\n```ts\ndeclare const tools: {{ {}(args: {}): Promise<{}>; }};\n```",
        identifier, render_arguments(definition), render_result(definition));
    return section;
}

#ifdef FILO_ENABLE_JAVASCRIPT

// What a script may collect for one result before truncation, so a loop
// calling text() cannot exhaust memory.
constexpr std::size_t kMaxCollectedBytes = 8 * 1024 * 1024;
constexpr std::size_t kMemoryLimitBytes = 64 * 1024 * 1024;
// Scripts run on tool threads, whose native stacks can be as small as 512 KiB.
constexpr std::size_t kMaxStackBytes = 256 * 1024;
constexpr int kMaxToolCalls = 256;
// What a script may attach with image() to one result: enough for
// screenshots, bounded so a loop cannot fill every later request with images.
constexpr std::size_t kMaxImages = 16;
constexpr std::size_t kMaxImageBytes = 20 * 1024 * 1024;
// Uninterrupted JavaScript between two waits is a runaway loop, not work.
constexpr auto kMaxSlice = std::chrono::seconds{30};
// Total time a script may sleep on its own timers.
constexpr auto kMaxTimerWait = std::chrono::seconds{60};
constexpr auto kCancelPoll = std::chrono::milliseconds{50};

// The script-facing helpers, written over the natives the realm installs.
// The natives leave the global object once the helpers hold them.
constexpr std::string_view kPrelude = R"((() => {
  const call = __filo_call, emit = __filo_text, put = __filo_store, get = __filo_load;
  const attach = __filo_image, note = __filo_notify;
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
  globalThis.notify = (value) => note(show(value));
  // A data: URL, { image_url }, or an MCP ImageContent block.
  const imageUrl = (input) => {
    if (typeof input === "string") return input;
    if (input && typeof input === "object") {
      if (typeof input.image_url === "string") return input.image_url;
      if (input.type === "image" && typeof input.data === "string") {
        return `data:${input.mimeType || "image/png"};base64,${input.data}`;
      }
    }
    throw new TypeError("image() takes a data: URL, { image_url }, or an MCP ImageContent block");
  };
  globalThis.image = (input) => attach(imageUrl(input));
  globalThis.generatedImage = (result) => {
    if (!result || typeof result.image_url !== "string") {
      throw new TypeError("generatedImage() takes { image_url, output_hint? }");
    }
    attach(result.image_url);
    if (result.output_hint !== undefined && result.output_hint !== null) emit(show(result.output_hint));
  };
  globalThis.store = (key, value) => put(String(key), JSON.stringify(value));
  globalThis.load = (key) => {
    const json = get(String(key));
    return json === undefined ? undefined : JSON.parse(json);
  };
  globalThis.exit = __filo_exit;
  globalThis.yield_control = __filo_yield;
  globalThis.setTimeout = __filo_set_timeout;
  globalThis.clearTimeout = __filo_clear_timeout;
  for (const name of ["__filo_call", "__filo_text", "__filo_store", "__filo_load",
                      "__filo_image", "__filo_notify",
                      "__filo_exit", "__filo_yield", "__filo_set_timeout", "__filo_clear_timeout",
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
    explicit ScriptRealm(ToolScriptStore& store)
        : store_(store), runtime_(JS_NewRuntime()) {
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
            if (promise_held_) JS_FreeValue(context_, promise_);
            for (auto& call : pending_) release(call);
            for (auto& [id, timer] : timers_) JS_FreeValue(context_, timer.callback);
            JS_FreeContext(context_);
        }
        if (runtime_ != nullptr) JS_FreeRuntime(runtime_);
    }

    void set_cell_id(std::string id) { cell_id_ = std::move(id); }
    [[nodiscard]] const std::string& cell_id() const noexcept { return cell_id_; }

    /// Starts the script. @p allow_yield keeps a still-running script parked
    /// in this realm; otherwise a yield point is not a place to stop.
    [[nodiscard]] ToolScriptOutcome run(std::string_view source,
                                        const std::vector<core::llm::Tool>& tools,
                                        ToolScriptHost& host,
                                        bool allow_yield) {
        const auto started = Clock::now();
        const auto pragma = parse_exec_pragma(source);
        budget_ = static_cast<std::size_t>(std::min<std::uint64_t>(
            pragma.max_output_tokens * kBytesPerToken, kMaxCollectedBytes));
        arm(host, allow_yield, pragma.yield_time_ms.value_or(kDefaultExecYieldMs));

        std::optional<std::string> error;
        if (context_ == nullptr) {
            error = "the JavaScript runtime did not start";
        } else if (!install(script_tool_surface(tools))) {
            error = stop_ == Stop::None
                ? "the script helpers did not start: " + take_exception()
                : failure_of_exception();
        } else {
            // JS_Eval reads one byte past the source: it must be terminated.
            const std::string program(source);
            begin_slice();
            OwnedValue evaluated(context_, JS_Eval(context_, program.c_str(), program.size(),
                                                   "exec", JS_EVAL_TYPE_GLOBAL | JS_EVAL_FLAG_ASYNC));
            if (evaluated.exception()) {
                error = failure_of_exception();
            } else {
                promise_ = JS_DupValue(context_, evaluated.get());
                promise_held_ = true;
            }
        }
        if (error) return finish(DriveKind::Failed, std::move(*error), started);
        const auto status = drive();
        return finish(status.kind, std::move(status.error), started);
    }

    [[nodiscard]] ToolScriptOutcome resume(ToolScriptHost& host,
                                           std::uint64_t yield_ms,
                                           std::optional<std::uint64_t> max_tokens) {
        const auto started = Clock::now();
        if (max_tokens && *max_tokens > 0) {
            budget_ = static_cast<std::size_t>(std::min<std::uint64_t>(
                *max_tokens * kBytesPerToken, kMaxCollectedBytes));
        }
        arm(host, true, yield_ms);
        const auto status = drive();
        return finish(status.kind, std::move(status.error), started);
    }

    [[nodiscard]] ToolScriptOutcome terminate() {
        const auto started = Clock::now();
        stop_ = Stop::Cancelled;
        return finish(DriveKind::Terminated, {}, started);
    }

private:
    enum class Stop { None, Cancelled, Runaway };
    enum class DriveKind { Done, Failed, Yielded, Terminated };

    struct DriveStatus {
        DriveKind kind = DriveKind::Done;
        std::string error;
    };

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
        if (self.host_ != nullptr && self.host_->cancelled()) {
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

    static JSValue native_notify(JSContext* context, JSValueConst, int argc, JSValueConst* argv) {
        auto& self = of(context);
        std::string note = argc > 0 ? to_text(context, argv[0]) : std::string("undefined");
        if (self.collected_ < kMaxCollectedBytes) {
            self.collected_ += note.size();
            self.notes_.push_back(std::move(note));
        }
        return JS_UNDEFINED;
    }

    static JSValue native_image(JSContext* context, JSValueConst, int argc, JSValueConst* argv) {
        auto& self = of(context);
        std::string url = argc > 0 ? to_text(context, argv[0]) : std::string{};
        const auto mime = image_data_url_mime(url);
        if (!mime) {
            return JS_ThrowTypeError(context,
                                     "image() needs a base64 data: URL of an image; HTTP(S) URLs "
                                     "are not supported");
        }
        if (self.images_.size() >= kMaxImages || self.image_bytes_ + url.size() > kMaxImageBytes) {
            return JS_ThrowRangeError(context, "the script attached too many images");
        }
        self.image_bytes_ += url.size();
        auto part = core::llm::ContentPart::make_image_url(std::move(url));
        part.mime_type = std::string(*mime);
        part.path = std::format("image {} attached by the exec script", self.images_.size() + 1);
        self.images_.push_back(std::move(part));
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

    static JSValue native_yield(JSContext* context, JSValueConst, int, JSValueConst*) {
        of(context).yield_requested_ = true;
        return JS_UNDEFINED;
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
            || !define("__filo_image", &native_image, 1)
            || !define("__filo_notify", &native_notify, 1)
            || !define("__filo_store", &native_store, 2) || !define("__filo_load", &native_load, 1)
            || !define("__filo_exit", &native_exit, 0)
            || !define("__filo_yield", &native_yield, 0)
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

    [[nodiscard]] bool should_yield() const noexcept {
        return allow_yield_ && (yield_requested_ || Clock::now() >= yield_at_);
    }

    void arm(ToolScriptHost& host, bool allow_yield, std::uint64_t yield_ms) {
        host_ = &host;
        allow_yield_ = allow_yield;
        yield_requested_ = false;
        stop_ = Stop::None;
        yield_at_ = allow_yield
            ? Clock::now() + std::chrono::milliseconds(yield_ms)
            : Clock::time_point::max();
    }

    /// The text items since the previous result, joined. What a result
    /// reported is released, and the collection caps start over, so a cell
    /// that runs for many waits neither keeps its old output nor stops
    /// reporting once its lifetime output passes them.
    [[nodiscard]] std::string take_new_items() {
        std::string items;
        for (const auto& item : items_) {
            if (!items.empty()) items += '\n';
            items += item;
        }
        items_.clear();
        collected_ = 0;
        return items;
    }

    [[nodiscard]] ToolScriptOutcome finish(DriveKind kind,
                                           std::string error,
                                           Clock::time_point started) {
        std::string items = take_new_items();
        std::string status;
        bool completed = false;
        bool running = false;
        if (kind == DriveKind::Yielded) {
            status = std::format("Script running with cell ID {}", cell_id_);
            running = true;
        } else if (kind == DriveKind::Done) {
            status = "Script completed";
            completed = true;
        } else if (kind == DriveKind::Terminated || stop_ == Stop::Cancelled) {
            status = "Script terminated";
        } else {
            status = "Script failed";
        }
        if (!error.empty() && kind != DriveKind::Yielded) {
            if (!items.empty()) items += '\n';
            items += "Script error:\n" + error;
        }
        host_ = nullptr;
        const auto notes = std::exchange(notes_, {});
        auto images = std::exchange(images_, {});
        image_bytes_ = 0;
        return ToolScriptOutcome{
            .completed = completed,
            .running = running,
            .output = render_output(status, Clock::now() - started, std::move(items), budget_,
                                    notes),
            .images = std::move(images),
        };
    }

    [[nodiscard]] DriveStatus drive() {
        for (;;) {
            if (auto failure = drain_jobs()) return {DriveKind::Failed, std::move(*failure)};
            if (exited_) return {};
            switch (JS_PromiseState(context_, promise_)) {
            case JS_PROMISE_PENDING:
                break;
            case JS_PROMISE_REJECTED: {
                OwnedValue reason(context_, JS_PromiseResult(context_, promise_));
                return {DriveKind::Failed, failure_of(reason.get()).value_or(std::string{})};
            }
            case JS_PROMISE_FULFILLED:
            case JS_PROMISE_NOT_A_PROMISE:
                return {};
            }
            if (host_ != nullptr && host_->cancelled()) {
                stop_ = Stop::Cancelled;
                return {DriveKind::Failed, failure_of(JS_UNDEFINED).value_or(std::string{})};
            }
            if (should_yield() && pending_.empty()) return {DriveKind::Yielded, {}};
            if (!pending_.empty()) {
                if (auto failure = settle_pending_calls()) {
                    return {DriveKind::Failed, std::move(*failure)};
                }
            } else if (!timers_.empty()) {
                const auto step = fire_next_timer();
                if (step.kind == DriveKind::Yielded) return step;
                if (step.kind == DriveKind::Failed) return step;
            } else {
                return {DriveKind::Failed, "the script awaited a promise that nothing settles"};
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
        for (auto& pending : batch) calls.push_back(std::move(pending.call));
        auto replies = host_->run(calls);

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

    [[nodiscard]] DriveStatus fire_next_timer() {
        const auto next = std::ranges::min_element(timers_, {}, [](const auto& entry) {
            return entry.second.due;
        });
        auto now = Clock::now();
        if (next->second.due > now) {
            const auto due_in = next->second.due - now;
            if (!allow_yield_ && timer_wait_ + due_in > kMaxTimerWait) {
                return {DriveKind::Failed,
                        std::format("the script waited on timers for more than {} seconds",
                                    kMaxTimerWait.count())};
            }
            const auto wake = allow_yield_ ? std::min(next->second.due, yield_at_) : next->second.due;
            while (now < wake) {
                if (host_ != nullptr && host_->cancelled()) {
                    stop_ = Stop::Cancelled;
                    return {DriveKind::Failed, failure_of(JS_UNDEFINED).value_or(std::string{})};
                }
                if (should_yield()) return {DriveKind::Yielded, {}};
                const auto pause = std::min<Clock::duration>(wake - now, kCancelPoll);
                std::this_thread::sleep_for(pause);
                if (!allow_yield_) timer_wait_ += pause;
                now = Clock::now();
            }
            if (now < next->second.due) return {DriveKind::Yielded, {}};
        }
        const JSValue callback = next->second.callback;
        timers_.erase(next);
        begin_slice();
        OwnedValue called(context_, JS_Call(context_, callback, JS_UNDEFINED, 0, nullptr));
        JS_FreeValue(context_, callback);
        if (called.exception()) {
            return {DriveKind::Failed, failure_of_exception().value_or(std::string{})};
        }
        return {};
    }

    ToolScriptHost* host_ = nullptr;
    ToolScriptStore& store_;
    JSRuntime* runtime_ = nullptr;
    JSContext* context_ = nullptr;
    JSValue promise_ = JS_UNDEFINED;
    bool promise_held_ = false;
    std::string cell_id_;
    std::vector<PendingCall> pending_;
    std::map<std::int64_t, Timer> timers_;
    std::int64_t next_timer_id_ = 0;
    Clock::duration timer_wait_{};
    Clock::time_point slice_started_ = Clock::now();
    Clock::time_point yield_at_ = Clock::time_point::max();
    std::vector<std::string> items_;
    std::vector<std::string> notes_;
    std::vector<core::llm::ContentPart> images_;
    std::size_t image_bytes_ = 0;
    std::size_t collected_ = 0;
    std::size_t budget_ = 0;
    int calls_ = 0;
    bool exited_ = false;
    bool allow_yield_ = false;
    bool yield_requested_ = false;
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
    // codex-rs code-mode-protocol/src/description.rs EXEC_DESCRIPTION_TEMPLATE
    // and EXEC_HELPERS_TEMPLATE, naming Filo's runtime, listing only the
    // helpers it implements, and adding how Filo answers a refused call.
    std::string description = std::format(
        "Run JavaScript code to orchestrate/compose tool calls\n"
        "- Evaluates the provided JavaScript code in a fresh QuickJS realm as an async script.\n"
        "- All nested tools are available on the global `tools` object, for example "
        "`await tools.exec_command(...)`. Tool names are exposed as normalized JavaScript "
        "identifiers, for example `await tools.mcp__ologs__get_profile(...)`.\n"
        "- Nested tool methods take either a string or an object as their input argument.\n"
        "- Nested tools return either an object or a string, based on the description.\n"
        "- Runs raw JavaScript -- no Node, no file system, no network access, no console.\n"
        "- Accepts raw JavaScript source text, not JSON, quoted strings, or markdown code fences.\n"
        "- You may optionally start the tool input with a first-line pragma like "
        "`// @exec: {{\"yield_time_ms\": 10000, \"max_output_tokens\": 1000}}`.\n"
        "- `yield_time_ms` asks `exec` to yield early if the script is still running. Defaults "
        "to {} ms.\n"
        "- `max_output_tokens` sets the token budget for direct `exec` results. Defaults to "
        "10000 tokens.\n"
        "- When the JS code is fully evaluated, the realm's lifetime ends and unawaited promises "
        "are silently discarded.\n"
        "- A nested call that cannot run (invalid arguments, blocked, or denied by the user) "
        "rejects with an Error whose message says why. Handle it in the script, and do not ask "
        "for the same permission again.\n"
        "- Nested calls awaited together run in parallel when they do not touch the same files.\n"
        "- Global helpers:\n"
        "- `exit()`: Immediately ends the current script successfully (like an early return from "
        "the top level).\n"
        "- `text(value: string | number | boolean | undefined | null)`: Appends a text item. "
        "Non-string values are stringified with `JSON.stringify(...)` when possible.\n"
        "- `image(imageUrlOrItem: string | {{ image_url: string }} | ImageContent)`: Appends an "
        "image item. `image_url` should be a base64-encoded `data:` URL. To forward an MCP tool "
        "image, pass an individual `ImageContent` block from `result.content`, for example "
        "`image(result.content[0])`.\n"
        "- `generatedImage(result: {{ image_url: string; output_hint?: string }})`: Appends an "
        "image-generation result and its optional output hint. HTTP(S) URLs are not supported.\n"
        "- `store(key: string, value: any)`: stores a serializable value under a string key for "
        "later `exec` calls in the same session.\n"
        "- `load(key: string)`: returns the stored value for a string key, or `undefined` if it "
        "is missing.\n"
        "- `notify(value: string | number | boolean | undefined | null)`: adds a message to this "
        "`exec` result that the output token budget never truncates. Values are stringified "
        "like `text(...)`.\n"
        "- `setTimeout(callback: () => void, delayMs?: number)`: schedules a callback to run "
        "later and returns a timeout id. Pending timeouts do not keep `exec` alive by themselves; "
        "await an explicit promise if you need to wait for one.\n"
        "- `clearTimeout(timeoutId?: number)`: cancels a timeout created by `setTimeout`.\n"
        "- `ALL_TOOLS`: metadata for the enabled nested tools as `{{ name, description }}` "
        "entries.\n"
        "- `yield_control()`: yields the accumulated output to the model immediately while the "
        "script keeps running.\n\n"
        "Shared MCP Types:\n```ts\n{}\n```",
        kDefaultExecYieldMs, kMcpTypescriptPreamble);
    for (const auto& tool : script_tool_surface(tools)) {
        description += "\n\n";
        description += render_tool_section(tool.function);
    }
    return description;
}

std::string wait_tool_description() {
    // codex-rs core/src/tools/code_mode/wait_spec.rs + WAIT_DESCRIPTION_TEMPLATE.
    return "Waits on a yielded `exec` cell and returns new output or completion.\n"
           "- Use `wait` only after `exec` returns `Script running with cell ID ...`.\n"
           "- `cell_id` identifies the running `exec` cell to resume.\n"
           "- `yield_time_ms` controls how long to wait for more output before yielding again. "
           "Defaults to 10000 ms.\n"
           "- `max_tokens` limits how much new output this wait call returns. Defaults to 10000 "
           "tokens.\n"
           "- `terminate: true` stops the running cell; false or omitted waits for output.\n"
           "- `wait` returns only the new output since the last yield, or the final completion or "
           "termination result for that cell.\n"
           "- If the cell is still running, `wait` may yield again with the same `cell_id`.\n"
           "- If the cell has already finished, `wait` returns the completed result and closes "
           "the cell.";
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

namespace {

// codex-rs unified_exec/mod.rs: yield bounds, and the defaults
// exec_command_handler and write_stdin use.
constexpr std::int64_t kMinYieldMs = 250;
constexpr std::int64_t kMaxYieldMs = 30'000;
constexpr std::int64_t kMinEmptyPollYieldMs = 5'000;
constexpr std::int64_t kMaxEmptyPollYieldMs = 300'000;
constexpr std::int64_t kDefaultCommandYieldMs = 10'000;
constexpr std::int64_t kDefaultWriteYieldMs = 250;

// codex-rs shell_spec.rs unified_exec_output_schema.
constexpr std::string_view kUnifiedExecOutputSchema =
    R"({"type":"object","properties":{)"
    R"("wall_time_seconds":{"type":"number","description":"Elapsed wall time spent waiting for output in seconds."},)"
    R"("exit_code":{"type":"number","description":"Process exit code when the command finished during this call."},)"
    R"("session_id":{"type":"number","description":"Session identifier to pass to write_stdin when the process is still running."},)"
    R"("original_token_count":{"type":"number","description":"Approximate token count before output truncation."},)"
    R"("output":{"type":"string","description":"Command output text, possibly truncated."}},)"
    R"("required":["wall_time_seconds","output"],"additionalProperties":false})";

constexpr std::string_view kMaxOutputTokensDescription =
    "Output token budget. Defaults to 10000 tokens; larger requests may be capped by policy.";

[[nodiscard]] std::optional<std::int64_t> integer_field(simdjson::dom::object object,
                                                        std::string_view key) {
    double value = 0;
    if (object[key].get(value) != simdjson::SUCCESS || !std::isfinite(value)) return std::nullopt;
    return static_cast<std::int64_t>(value);
}

} // namespace

ToolDefinition exec_command_definition() {
    return ToolDefinition{
        .name = std::string(names::kExecCommand),
        .title = "Run",
        .description =
            "Runs a command in a PTY, returning output or a session ID for ongoing interaction.",
        .parameters = {
            {"cmd", "string", "Shell command to execute.", true},
            {"workdir", "string", "Working directory for the command. Defaults to the turn cwd.",
             false},
            {"tty", "boolean",
             "True allocates a PTY for the command; false or omitted uses plain pipes.", false},
            {"yield_time_ms", "number",
             "Wait before yielding output. Defaults to 10000 ms; effective range is 250-30000 ms.",
             false},
            {"max_output_tokens", "number", std::string(kMaxOutputTokensDescription), false},
        },
        .output_schema = std::string(kUnifiedExecOutputSchema),
        .annotations = {.destructive_hint = true, .open_world_hint = true},
    };
}

ToolDefinition write_stdin_definition() {
    return ToolDefinition{
        .name = std::string(names::kWriteStdin),
        .title = "Run",
        .description =
            "Writes characters to an existing unified exec session and returns recent output.",
        .parameters = {
            {"session_id", "number", "Identifier of the running unified exec session.", true},
            {"chars", "string",
             "Bytes to write to stdin. Defaults to empty, which polls without writing.", false},
            {"yield_time_ms", "number",
             "Wait before yielding output. Non-empty writes default to 250 ms and cap at 30000 "
             "ms; empty polls wait 5000-300000 ms by default.",
             false},
            {"max_output_tokens", "number", std::string(kMaxOutputTokensDescription), false},
        },
        .output_schema = std::string(kUnifiedExecOutputSchema),
        .annotations = {.destructive_hint = true, .open_world_hint = true},
    };
}

bool is_direct_only_tool(std::string_view tool_name) noexcept {
    return tool_name == names::kAskUserQuestion
        || tool_name == names::kRequestUserInputAsync
        || tool_name == names::kSendMessageToUserAsync;
}

std::vector<core::llm::Tool> script_tool_surface(const std::vector<core::llm::Tool>& offered) {
    std::vector<core::llm::Tool> surface;
    surface.reserve(offered.size() + 1);
    for (const auto& tool : offered) {
        if (is_direct_only_tool(tool.function.name)) continue;
        if (tool.function.name == names::kRunTerminalCommand) {
            surface.push_back(core::llm::Tool{.function = exec_command_definition()});
            surface.push_back(core::llm::Tool{.function = write_stdin_definition()});
            continue;
        }
        surface.push_back(tool);
    }
    return surface;
}

std::optional<ToolDefinition> script_alias_definition(std::string_view tool_name) {
    if (tool_name == names::kExecCommand) return exec_command_definition();
    if (tool_name == names::kWriteStdin) return write_stdin_definition();
    return std::nullopt;
}

std::optional<ScriptToolAlias> resolve_script_alias(std::string_view tool_name,
                                                    std::string_view arguments,
                                                    std::string_view default_workdir) {
    const bool command = tool_name == names::kExecCommand;
    if (!command && tool_name != names::kWriteStdin) return std::nullopt;
    ScriptToolAlias alias{
        .tool_name = std::string(names::kRunTerminalCommand),
        .arguments = "{}",
        .max_output_tokens = kDefaultMaxOutputTokens,
        .pre_approved = !command,
    };
    simdjson::dom::parser parser;
    const simdjson::padded_string padded(arguments);
    simdjson::dom::object object;
    if (parser.parse(padded).get(object) != simdjson::SUCCESS) return alias;
    if (const auto tokens = integer_field(object, "max_output_tokens"); tokens && *tokens >= 1) {
        alias.max_output_tokens = static_cast<std::uint64_t>(*tokens);
    }
    const auto yield = integer_field(object, "yield_time_ms");

    if (command) {
        std::string_view cmd;
        if (object["cmd"].get(cmd) != simdjson::SUCCESS) return alias;
        std::string_view workdir = default_workdir;
        if (std::string_view dir; object["workdir"].get(dir) == simdjson::SUCCESS && !dir.empty()) {
            workdir = dir;
        }
        alias.arguments = R"({"command":")" + core::utils::escape_json_string(cmd) + '"';
        if (!workdir.empty()) {
            alias.arguments += R"(,"working_dir":")" + core::utils::escape_json_string(workdir) + '"';
        }
        if (bool tty = false; object["tty"].get(tty) == simdjson::SUCCESS && tty) {
            alias.arguments += R"(,"tty":true)";
        }
        alias.arguments += std::format(
            R"(,"yield_time_ms":{}}})",
            std::clamp(yield.value_or(kDefaultCommandYieldMs), kMinYieldMs, kMaxYieldMs));
        return alias;
    }

    const auto session = integer_field(object, "session_id");
    if (!session) return alias;
    std::string_view chars;
    std::ignore = object["chars"].get(chars);
    const std::int64_t yield_ms = chars.empty()
        ? std::clamp(yield.value_or(kMinEmptyPollYieldMs), kMinEmptyPollYieldMs, kMaxEmptyPollYieldMs)
        : std::clamp(yield.value_or(kDefaultWriteYieldMs), kMinYieldMs, kMaxYieldMs);
    alias.arguments = std::format(R"({{"process_id":{},"input":"{}","yield_time_ms":{}}})",
                                  *session, core::utils::escape_json_string(chars), yield_ms);
    return alias;
}

std::string script_alias_result(const ScriptToolAlias& alias, std::string_view result) {
    simdjson::dom::parser parser;
    const simdjson::padded_string padded(result);
    simdjson::dom::object document;
    std::string_view output;
    if (parser.parse(padded).get(document) != simdjson::SUCCESS
        || document["output"].get(output) != simdjson::SUCCESS) {
        return std::string(result);
    }
    std::optional<std::int64_t> exit_code;
    if (std::int64_t code = 0; document["exit_code"].get(code) == simdjson::SUCCESS) exit_code = code;
    std::optional<std::int64_t> session;
    if (std::int64_t id = 0; document["process_id"].get(id) == simdjson::SUCCESS) session = id;
    if (!exit_code && !session) return std::string(result);
    double wall_time_seconds = 0;
    std::ignore = document["wall_time_seconds"].get(wall_time_seconds);

    // codex-rs utils/string truncate.rs plan_token_truncation: keep the head
    // and the tail of the budget, and say how many tokens were cut.
    const std::size_t budget = alias.max_output_tokens * kBytesPerToken;
    std::string text;
    std::optional<std::uint64_t> original_tokens;
    if (output.size() <= budget) {
        text = std::string(output);
    } else {
        const auto head = utf8_prefix(output, budget / 2);
        const auto tail = utf8_suffix(output, budget - budget / 2);
        const auto removed = output.size() - budget;
        text = std::format("{}…{} tokens truncated…{}", head,
                           (removed + kBytesPerToken - 1) / kBytesPerToken, tail);
        original_tokens = (output.size() + kBytesPerToken - 1) / kBytesPerToken;
    }

    std::string json = std::format(R"({{"wall_time_seconds":{:.3f})", wall_time_seconds);
    if (exit_code) json += std::format(R"(,"exit_code":{})", *exit_code);
    if (session) json += std::format(R"(,"session_id":{})", *session);
    if (original_tokens) json += std::format(R"(,"original_token_count":{})", *original_tokens);
    json += R"(,"output":")" + core::utils::escape_json_string(text) + "\"}";
    return json;
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
    ScriptRealm realm(store);
    return realm.run(source, tools, host, /*allow_yield=*/false);
#else
    (void)tools;
    (void)host;
    (void)store;
    (void)source;
    return ToolScriptOutcome{
        .completed = false,
        .output = render_output("Script failed", {},
                                "Script error:\nthis build of Filo has no JavaScript runtime",
                                static_cast<std::size_t>(kDefaultMaxOutputTokens * kBytesPerToken)),
    };
#endif
}

namespace {

constexpr std::size_t kMaxExecCells = 8;

[[nodiscard]] ToolScriptOutcome wait_refusal(std::string text) {
    return ToolScriptOutcome{.output = std::move(text)};
}

struct WaitArguments {
    std::string cell_id;
    std::uint64_t yield_ms = kDefaultExecYieldMs;
    std::optional<std::uint64_t> max_tokens;
    bool terminate = false;
    std::string error;
};

[[nodiscard]] WaitArguments parse_wait_arguments(std::string_view arguments) {
    WaitArguments parsed;
    simdjson::dom::parser parser;
    const simdjson::padded_string padded(arguments);
    simdjson::dom::element document;
    std::string_view cell_id;
    if (parser.parse(padded).get(document) != simdjson::SUCCESS
        || document["cell_id"].get(cell_id) != simdjson::SUCCESS
        || cell_id.empty()) {
        parsed.error = "wait needs a cell_id. exec returns \"Script running with cell ID ...\" "
                       "while a script is still going.";
        return parsed;
    }
    parsed.cell_id.assign(cell_id);
    std::uint64_t yield_ms = 0;
    if (document["yield_time_ms"].get(yield_ms) == simdjson::SUCCESS) parsed.yield_ms = yield_ms;
    std::uint64_t max_tokens = 0;
    if (document["max_tokens"].get(max_tokens) == simdjson::SUCCESS && max_tokens > 0) {
        parsed.max_tokens = max_tokens;
    }
    bool terminate = false;
    if (document["terminate"].get(terminate) == simdjson::SUCCESS) parsed.terminate = terminate;
    return parsed;
}

} // namespace

struct ToolScriptSession::State {
#ifdef FILO_ENABLE_JAVASCRIPT
    ToolScriptStore store;
    std::vector<std::unique_ptr<ScriptRealm>> cells;
    int next_id = 0;
#endif
};

ToolScriptSession::ToolScriptSession() : state_(std::make_unique<State>()) {}

ToolScriptSession::~ToolScriptSession() = default;

void ToolScriptSession::clear() {
#ifdef FILO_ENABLE_JAVASCRIPT
    state_->cells.clear();
    state_->store.clear();
#else
    (void)state_;
#endif
}

ToolScriptOutcome ToolScriptSession::execute(std::string_view source,
                                             const std::vector<core::llm::Tool>& tools,
                                             ToolScriptHost& host) {
#ifdef FILO_ENABLE_JAVASCRIPT
    if (state_->cells.size() >= kMaxExecCells) {
        return wait_refusal(
            "Too many exec scripts are still running. wait for one to finish, or terminate it, "
            "before starting another.");
    }
    const std::string id = "c" + std::to_string(++state_->next_id);
    auto realm = std::make_unique<ScriptRealm>(state_->store);
    realm->set_cell_id(id);
    auto outcome = realm->run(source, tools, host, /*allow_yield=*/true);
    if (outcome.running) state_->cells.push_back(std::move(realm));
    return outcome;
#else
    (void)source;
    (void)tools;
    (void)host;
    return wait_refusal("Script error:\nthis build of Filo has no JavaScript runtime");
#endif
}

ToolScriptOutcome ToolScriptSession::resume(std::string_view arguments, ToolScriptHost& host) {
    const auto parsed = parse_wait_arguments(arguments);
    if (!parsed.error.empty()) return wait_refusal(parsed.error);
#ifdef FILO_ENABLE_JAVASCRIPT
    const auto found = std::ranges::find_if(state_->cells, [&](const auto& cell) {
        return cell && cell->cell_id() == parsed.cell_id;
    });
    if (found == state_->cells.end()) {
        return wait_refusal(std::format(
            "No running exec cell \"{}\". exec returns \"Script running with cell ID {}\" only "
            "while that script is still going.",
            parsed.cell_id, parsed.cell_id));
    }
    auto outcome = parsed.terminate
        ? (*found)->terminate()
        : (*found)->resume(host, parsed.yield_ms, parsed.max_tokens);
    if (!outcome.running) state_->cells.erase(found);
    return outcome;
#else
    (void)host;
    return wait_refusal("Script error:\nthis build of Filo has no JavaScript runtime");
#endif
}

} // namespace core::tools
