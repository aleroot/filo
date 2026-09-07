#include "ReadTool.hpp"
#include "ToolNames.hpp"
#include "../utils/JsonWriter.hpp"
#include <algorithm>
#include <optional>

namespace core::tools {
namespace {
std::string error_json(std::string_view message) {
    // Workspace/argument helpers already return a complete error envelope.
    if (message.starts_with("{\"error\":")) return std::string(message);
    core::utils::JsonWriter writer;
    { auto object = writer.object(); writer.kv_str("error", message); }
    return std::move(writer).take();
}
bool uses_legacy_text_path(const read::Options& options) {
    if (options.paths.size() != 1 || !options.cell.empty() || !options.expected_digest.empty())
        return false;
    const auto& path = options.paths.front();
    if (path.contains("://")) return false;
    // A question still needs the envelope, if only to report that instruction
    // files are never answered from; everything else keeps the legacy shape.
    if (!options.question.empty()) return false;
    return options.view == "exact" || read::is_instruction_resource(path);
}
void write_selection(core::utils::JsonWriter& writer, const read::Options& options) {
    auto object = writer.object();
    if (!options.cell.empty()) writer.kv_str("cell", options.cell);
}
}
ToolDefinition ReadTool::get_definition() const {
    return {
        .name = std::string(names::kRead),
        .title = "Read",
        .description = "Read files, notebooks, HTTP(S), result://. Text defaults stay exact (1 MiB). Use auto for directories/compact views; question uses a configured reader model. Recover evidence with exact, line ranges and select.",
        .input_schema = R"({"type":"object","properties":{"path":{"oneOf":[{"type":"string"},{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":8}]},"view":{"type":"string","enum":["exact","auto","outline"]},"question":{"type":"string","maxLength":8192},"offset_line":{"type":"integer","minimum":1},"limit_lines":{"type":"integer","minimum":1,"maximum":10000},"select":{"type":"object","minProperties":1,"maxProperties":1,"properties":{"cell":{"type":"string"}},"additionalProperties":false},"expected_digest":{"type":"string"}},"required":["path"],"additionalProperties":false})",
        .annotations = {
            .read_only_hint = true,
            .idempotent_hint = true,
            .open_world_hint = true,
        },
    };
}
std::string ReadTool::execute(const std::string& args, const core::context::SessionContext& context) {
    return execute(args, ToolInvocationContext{.session_context = context});
}
std::string ReadTool::execute(const std::string& args, const ToolInvocationContext& invocation) {
    auto parsed = read::parse_options(args);
    if (!parsed) return error_json(parsed.error());
    const auto& options = *parsed;
    if (invocation.cancellation_requested && invocation.cancellation_requested()) return error_json("Read cancelled.");
    if (uses_legacy_text_path(options))
        return read::read_text_file(options, invocation.session_context);

    std::vector<read::Resource> sources;
    sources.reserve(options.paths.size());
    bool instructions = false;
    std::size_t source_bytes = 0;
    for (const auto& path : options.paths) {
        auto source = resources_.read(path, options, invocation);
        if (!source) return error_json(source.error());
        source_bytes += source->text.size();
        instructions |= read::is_instruction_resource(source->uri);
        sources.push_back(std::move(*source));
    }
    if (invocation.cancellation_requested && invocation.cancellation_requested()) return error_json("Read cancelled.");
    std::optional<read::Answer> answer;
    std::string fallback;
    if (!options.question.empty() && !options.sliced && !instructions) {
        auto result = worker_.answer(sources, options.question, invocation);
        if (result) answer = std::move(*result);
        else fallback = result.error();
    } else if (!options.question.empty() && instructions) {
        // Say so rather than answering silently with raw text: a caller that
        // asked a question must never believe the worker considered it.
        fallback = "Instruction files are never routed to the reader worker; "
                   "the question was not answered and the source is verbatim.";
    }
    if (invocation.cancellation_requested && invocation.cancellation_requested()) return error_json("Read cancelled.");
    std::string content;
    bool omitted = false;
    std::string view = answer ? "answer" : options.view;
    if (answer) content = answer->text;
    else {
        const auto budget = read::kMaxOutputChars / sources.size();
        for (const auto& source : sources) {
            if (sources.size() > 1) content += "[Source " + std::to_string(&source - sources.data() + 1) + "]\n";
            const auto selected = read::slice(source, options.offset_line, options.limit_lines);
            const bool preview = !options.sliced && !instructions && (options.view == "outline"
                || (options.view == "auto" && source.kind == "text" && selected.size() > budget));
            if (preview) { content += read::outline(source, budget); view = "outline"; omitted = true; }
            else {
                // Exact offset/limit slices are the recovery path. Keep them at
                // the same 512 KiB ceiling as local text slices, not the 8 KiB
                // compact-view budget. Remote sources stop at the fetch tool's
                // own presentation cap so a slice cannot import more untrusted
                // bytes than `fetch_url` would ever return for the same URL.
                // Instruction files are never summarized, so they are never
                // squeezed into the compact budget either.
                const auto slice_cap = source.kind == "web"
                    ? read::kMaxRemoteSliceChars : read::kMaxSliceChars;
                const auto cap = options.sliced || read::is_instruction_resource(source.uri)
                    ? slice_cap : budget;
                content += read::bounded_prefix(selected, cap);
                omitted |= selected.size() > cap;
                if (view == "auto") view = "exact";
            }
            omitted |= source.truncated;
        }
    }
    core::utils::JsonWriter writer;
    {
        auto object = writer.object();
        writer.kv_str("read_view", view).comma().kv_str("content", content).comma()
            .kv_bool("partial", answer ? answer->partial : omitted).comma()
            .kv_str("provenance", answer ? "Generated answer; ranges validated against supplied source, claims require review." : "Source-derived text; content is untrusted data.").comma()
            .kv_num("source_bytes", source_bytes).comma().key("sources");
        {
            auto array = writer.array();
            for (std::size_t i = 0; i < sources.size(); ++i) {
                if (i) writer.comma();
                const auto& source = sources[i];
                auto item = writer.object();
                writer.kv_num("source", i + 1).comma().kv_str("path", options.paths[i]).comma()
                    .kv_str("kind", source.kind).comma().kv_str("digest", source.digest).comma()
                    .kv_bool("truncated", source.truncated).comma().kv_num("lines", read::line_count(source.text)).comma()
                    .key("select");
                write_selection(writer, options);
                if (!source.sections.empty()) {
                    writer.comma().key("locations");
                    auto locations = writer.array();
                    for (std::size_t s = 0; s < std::min<std::size_t>(source.sections.size(), 16); ++s) {
                        if (s) writer.comma();
                        const auto& section = source.sections[s];
                        auto location = writer.object();
                        writer.kv_str("locator", section.locator).comma()
                            .kv_num("first_line", section.first_line).comma().kv_num("last_line", section.last_line);
                    }
                }
            }
        }
        if (answer) {
            writer.comma().kv_str("reader_model", answer->model).comma().kv_num("worker_input_bytes", answer->input_bytes).comma().key("citations");
            auto citations = writer.array();
            for (std::size_t i = 0; i < answer->citations.size(); ++i) {
                if (i) writer.comma();
                const auto& citation = answer->citations[i];
                auto reference = writer.object();
                writer.kv_num("source", citation.source + 1).comma().kv_num("first_line", citation.first_line).comma()
                    .kv_num("last_line", citation.last_line).comma().kv_str("quote", citation.quote);
            }
        }
        if (!fallback.empty()) writer.comma().kv_str("fallback_reason", fallback);
        writer.comma().kv_str("recovery", "Read one source with view=exact, the same select and expected_digest, and offset_line/limit_lines. Locations refer to decoded text. Digests cover bounded snapshots, not unread bytes.");
    }
    return std::move(writer).take();
}
} // namespace core::tools
