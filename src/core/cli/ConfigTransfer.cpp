#include "ConfigTransfer.hpp"

#include "core/utils/InterprocessFile.hpp"

#include <algorithm>
#include <format>
#include <iostream>
#include <iterator>
#include <ranges>

namespace core::cli {
namespace {

using namespace core::portability;

constexpr std::string_view kStdio = "-";

[[nodiscard]] std::size_t kind_column_width() {
    return std::ranges::max(item_kinds() | std::views::transform([](const KindInfo& info) {
        return info.name.size();
    }));
}

[[nodiscard]] std::string item_summary(const BundleItem& item) {
    std::string summary;
    if (item.kind == ItemKind::Memory) {
        const auto* entries = item.data.find("entries");
        const auto count = entries && entries->is_array() ? entries->items().size() : 0;
        summary = std::format("{} entr{}", count, count == 1 ? "y" : "ies");
    }
    if (!item.redacted.empty()) {
        if (!summary.empty()) summary += "; ";
        summary += "withheld: ";
        summary.append_range(item.redacted | std::views::join_with(std::string_view(", ")));
    }
    return summary;
}

void print_line(std::ostream& stream, std::string_view kind, std::string_view name,
                std::string_view detail) {
    stream << std::format("  {:<{}}  {}", kind, kind_column_width(), name);
    if (!detail.empty()) stream << "  (" << detail << ')';
    stream << '\n';
}

void print_warnings(std::ostream& stream, const std::vector<std::string>& warnings) {
    for (const auto& warning : warnings) stream << "warning: " << warning << '\n';
}

[[nodiscard]] std::expected<ItemFilter, std::string> request_filter(
    const ConfigTransferRequest& request) {
    return ItemFilter::parse(request.only, request.except);
}

int run_export(const ConfigTransferRequest& request, const PortabilityPaths& paths,
               std::ostream& out, std::ostream& err) {
    auto filter = request_filter(request);
    if (!filter) {
        err << "--only/--except: " << filter.error() << '\n';
        return 2;
    }
    const std::string target = request.export_target.value_or("");
    const bool to_stdout = target == kStdio;
    if (target.empty() && !request.dry_run) {
        err << "--export needs a file path, or - for standard output.\n";
        return 2;
    }

    auto result = export_bundle(paths, {.filter = std::move(*filter),
                                        .include_secrets = request.include_secrets});
    auto& report = to_stdout ? err : out;
    const auto& items = result.bundle.items;
    const bool compress =
        request.compress || (!to_stdout && bundle_encoding_for(target) == BundleEncoding::Gzip);
    const auto encoding = compress ? BundleEncoding::Gzip : BundleEncoding::Json;

    if (!request.dry_run) {
        if (to_stdout && request.stdout_is_terminal) {
            if (result.bundle.contains_secrets()) {
                err << "Refusing to print credentials to the terminal. "
                       "Redirect the output or pass a file path to --export.\n";
                return 2;
            }
            if (encoding == BundleEncoding::Gzip) {
                err << "Refusing to print a compressed bundle to the terminal. "
                       "Redirect the output or pass a file path to --export.\n";
                return 2;
            }
        }
        const std::string bytes = encode_bundle(result.bundle, encoding);
        if (to_stdout) {
            out << bytes << std::flush;
        } else {
            // A configuration snapshot is private whether or not it carries
            // credentials: it maps out accounts, endpoints and projects.
            std::string error;
            if (!core::utils::atomic_write_file(target, bytes, &error,
                                                core::utils::kPrivateFilePermissions)) {
                err << "Cannot write " << target << ": " << error << '\n';
                return 1;
            }
        }
    }

    if (items.empty()) {
        report << "No configuration matched; "
               << (request.dry_run ? "nothing would be exported" : "the bundle is empty") << ".\n";
    } else {
        report << std::format("{} {} item(s){}{}:\n",
                              request.dry_run ? "Would export" : "Exported", items.size(),
                              request.dry_run || to_stdout ? "" : " to " + target,
                              encoding == BundleEncoding::Gzip ? " (gzip-compressed)" : "");
        for (const auto& item : items) {
            print_line(report, kind_info(item.kind).name, item.name, item_summary(item));
        }
    }
    if (!result.withheld.empty()) {
        report << std::format("Withheld {} secret item(s):", result.withheld.size());
        for (const auto& name : result.withheld) report << ' ' << name;
        report << "\n  Add --include-secrets to carry OAuth logins and API keys.\n";
    }
    if (result.bundle.contains_secrets() && !request.dry_run) {
        report << "This bundle contains credentials: anyone holding it can use your accounts. "
                  "Move it over a private channel and delete it after importing.\n";
    }
    print_warnings(report, result.warnings);
    return 0;
}

int run_import(const ConfigTransferRequest& request, const PortabilityPaths& paths,
               std::istream& in, std::ostream& out, std::ostream& err) {
    auto filter = request_filter(request);
    if (!filter) {
        err << "--only/--except: " << filter.error() << '\n';
        return 2;
    }
    std::vector<PathMapping> mappings;
    for (const auto& spec : request.map_paths) {
        auto mapping = parse_path_mapping(spec);
        if (!mapping) {
            err << mapping.error() << '\n';
            return 2;
        }
        mappings.push_back(std::move(*mapping));
    }

    const std::string source = request.import_source.value_or(std::string(kStdio));
    std::string text;
    if (source == kStdio) {
        text.assign(std::istreambuf_iterator<char>(in), {});
    } else if (auto content = core::utils::read_file(source)) {
        text = std::move(*content);
    } else {
        err << "Cannot read " << source << ".\n";
        return 1;
    }
    auto bundle = decode_bundle(text);
    if (!bundle) {
        err << "Cannot import " << source << ": " << bundle.error() << '\n';
        return 1;
    }

    const auto result = import_bundle(*bundle, paths, {.filter = std::move(*filter),
                                                       .dry_run = request.dry_run,
                                                       .path_mappings = std::move(mappings)});
    if (result.items.empty()) {
        out << (bundle->items.empty() ? "The bundle is empty.\n"
                                      : "No item in the bundle matches the filter.\n");
        return 0;
    }
    out << std::format("{} {} item(s) from {}{}:\n",
                       request.dry_run ? "Would import" : "Imported", result.items.size(),
                       source == kStdio ? "standard input" : source,
                       bundle->created_at.empty() ? "" : " (exported " + bundle->created_at + ")");
    for (const auto& item : result.items) {
        std::string detail(import_action_name(item.action));
        if (!item.detail.empty()) detail += ": " + item.detail;
        print_line(out, kind_info(item.kind).name, item.name, detail);
    }
    if (result.backup) {
        out << "Replaced items were saved to " << result.backup->string()
            << "\n  Restore them with: filo --import " << result.backup->string() << '\n';
    }
    if (request.dry_run) out << "Dry run: nothing was written.\n";
    print_warnings(out, result.warnings);
    if (!result.ok()) {
        err << (!result.write_started ? "Import failed; nothing was written.\n"
                                : "Import failed part-way; see the items above.\n");
        return 1;
    }
    return 0;
}

} // namespace

int run_config_transfer(const ConfigTransferRequest& request,
                        const PortabilityPaths& paths,
                        std::istream& in,
                        std::ostream& out,
                        std::ostream& err) {
    if (request.export_target.has_value() == request.import_source.has_value()) {
        err << "Use exactly one of --export and --import.\n";
        return 2;
    }
    return request.export_target ? run_export(request, paths, out, err)
                                 : run_import(request, paths, in, out, err);
}

std::string config_transfer_selector_help() {
    std::string help =
        "Select items by KIND, KIND:PATTERN or a bare PATTERN matching any kind's item names "
        "(globs: * and ?). Kinds:";
    for (const auto& info : item_kinds()) {
        help += std::format(" {}{} ({})", info.name, info.secret ? "*" : "", info.description);
        help += info.kind == item_kinds().back().kind ? "." : ";";
    }
    help += " * = exported only with --include-secrets.";
    return help;
}

} // namespace core::cli
