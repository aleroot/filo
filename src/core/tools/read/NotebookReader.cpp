#include "ReadTypes.hpp"
#include "../../utils/JsonUtils.hpp"
#include <simdjson.h>
#include <format>
#include <algorithm>
#include <charconv>

namespace core::tools::read {
std::expected<Resource, std::string> decode_notebook(Resource resource, const Options& options) {
    if (resource.truncated) return std::unexpected("Notebook exceeds the 2 MiB input budget.");
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    simdjson::dom::array cells;
    if (parser.parse(resource.text).get(doc) || doc["cells"].get(cells))
        return std::unexpected("Invalid notebook: expected a cells array.");
    std::string output;
    std::vector<Section> sections;
    std::size_t index = 0, line = 1;
    bool found = options.cell.empty();
    std::size_t selected_index = 0;
    const auto [end, number_error] = std::from_chars(options.cell.data(), options.cell.data() + options.cell.size(), selected_index);
    const bool numeric = !options.cell.empty() && number_error == std::errc{} && end == options.cell.data() + options.cell.size();
    if (cells.size() > 10000) return std::unexpected("Notebook exceeds the 10000-cell budget.");
    for (auto cell : cells) {
        ++index;
        std::string_view id, type;
        core::utils::json::ignore_error(cell["id"].get(id));
        if (cell["cell_type"].get(type)) return std::unexpected("Invalid notebook cell type.");
        if (!options.cell.empty() && (numeric ? selected_index != index : options.cell != id)) continue;
        if (!options.cell.empty() && found) return std::unexpected("Notebook contains ambiguous cell ids.");
        found = true;
        const std::string locator = std::to_string(index);
        std::string text = std::format("[cell {}{}: {}]\n", index,
            id.empty() ? "" : std::format(" id={}", id), type);
        simdjson::dom::element source;
        if (cell["source"].get(source)) return std::unexpected("Notebook cell has no source.");
        std::string_view part;
        if (!source.get(part)) text += part;
        else {
            simdjson::dom::array parts;
            if (source.get(parts)) return std::unexpected("Invalid notebook cell source.");
            for (auto item : parts) {
                if (item.get(part)) return std::unexpected("Invalid notebook source fragment.");
                text += part;
            }
        }
        if (!text.ends_with('\n')) text += '\n';
        if (output.size() + text.size() > kMaxSourceBytes) return std::unexpected("Decoded notebook exceeds the 2 MiB budget.");
        const auto count = static_cast<std::size_t>(std::ranges::count(text, '\n'));
        sections.push_back({line, line + count - 1, "cell=" + locator});
        line += count;
        output += text;
    }
    if (!found) return std::unexpected("Notebook cell was not found (use its id or 1-based index).");
    resource.kind = "notebook";
    resource.text = std::move(output);
    resource.sections = std::move(sections);
    return resource;
}
} // namespace core::tools::read
