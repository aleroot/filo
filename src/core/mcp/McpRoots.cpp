#include "McpRoots.hpp"
#include "../utils/AsciiUtils.hpp"
#include "../utils/JsonWriter.hpp"
#include "../utils/UriUtils.hpp"

#include <simdjson.h>

#include <filesystem>
#include <format>
#include <system_error>

namespace core::mcp {
namespace {

using core::utils::JsonWriter;

[[nodiscard]] std::filesystem::path normalize_path(const std::filesystem::path& path) {
    std::error_code ec;
    auto normalized = std::filesystem::weakly_canonical(path, ec);
    if (!ec) {
        return normalized.lexically_normal();
    }

    ec.clear();
    normalized = std::filesystem::absolute(path, ec);
    if (!ec) {
        return normalized.lexically_normal();
    }

    return path.lexically_normal();
}

[[nodiscard]] std::optional<std::filesystem::path>
parse_local_file_uri(std::string_view uri_value, std::string& error_out) {
    if (!core::utils::ascii::istarts_with(uri_value, "file://")) {
        error_out = "Only file:// root URIs are supported";
        return std::nullopt;
    }

    std::string_view rest = uri_value.substr(7);
    std::string_view encoded_path;
    if (rest.starts_with('/')) {
        encoded_path = rest;
    } else {
        const auto slash = rest.find('/');
        const std::string_view authority =
            slash == std::string_view::npos ? rest : rest.substr(0, slash);
        if (!authority.empty() && !core::utils::ascii::iequals(authority, "localhost")) {
            error_out = "Unsupported file URI authority";
            return std::nullopt;
        }
        encoded_path = slash == std::string_view::npos ? std::string_view{"/"} : rest.substr(slash);
    }

    if (encoded_path.find('?') != std::string_view::npos
        || encoded_path.find('#') != std::string_view::npos) {
        error_out = "File URI must not include query or fragment";
        return std::nullopt;
    }

    std::string decoded_path;
    if (!core::utils::uri::percent_decode(encoded_path, decoded_path)) {
        error_out = "Malformed percent-encoding in file URI";
        return std::nullopt;
    }

    if (decoded_path.empty() || decoded_path.find('\0') != std::string::npos) {
        error_out = "Invalid file URI path";
        return std::nullopt;
    }

    return normalize_path(std::filesystem::path(decoded_path));
}

} // namespace

RootsCapability extract_roots_capability_from_initialize(std::string_view json_body) {
    simdjson::ondemand::parser parser;
    simdjson::padded_string padded(json_body);
    simdjson::ondemand::document doc;
    if (parser.iterate(padded).get(doc) != simdjson::SUCCESS) return {};

    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) return {};

    simdjson::ondemand::object params;
    if (root["params"].get_object().get(params) != simdjson::SUCCESS) return {};

    simdjson::ondemand::object capabilities;
    if (params["capabilities"].get_object().get(capabilities) != simdjson::SUCCESS) {
        return {};
    }

    simdjson::ondemand::object roots;
    if (capabilities["roots"].get_object().get(roots) != simdjson::SUCCESS) {
        return {};
    }

    bool list_changed = false;
    [[maybe_unused]] const auto ignored = roots["listChanged"].get(list_changed);
    return {.supported = true, .list_changed = list_changed};
}

bool is_workspace_sensitive_request(std::string_view method) {
    return method == "tools/call"
        || method == "resources/list"
        || method == "resources/read";
}

std::string build_roots_list_request_body(std::string_view request_id) {
    JsonWriter w(96);
    {
        auto _obj = w.object();
        w.kv_str("jsonrpc", "2.0").comma()
            .kv_str("id", request_id).comma()
            .kv_str("method", "roots/list");
    }
    return std::move(w).take();
}

std::optional<core::workspace::WorkspaceSnapshot>
parse_roots_list_response(std::string_view response_body,
                          std::string_view expected_request_id,
                          std::uint64_t workspace_version,
                          std::string& error_out) {
    simdjson::ondemand::parser parser;
    simdjson::padded_string padded(response_body);
    simdjson::ondemand::document doc;
    if (parser.iterate(padded).get(doc) != simdjson::SUCCESS) {
        error_out = "Client returned invalid JSON for roots/list";
        return std::nullopt;
    }

    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) {
        error_out = "Client returned invalid JSON-RPC response for roots/list";
        return std::nullopt;
    }

    std::string_view jsonrpc;
    if (root["jsonrpc"].get_string().get(jsonrpc) != simdjson::SUCCESS || jsonrpc != "2.0") {
        error_out = "Client roots/list response used an invalid JSON-RPC version";
        return std::nullopt;
    }

    std::string_view response_id;
    if (root["id"].get_string().get(response_id) != simdjson::SUCCESS
        || response_id != expected_request_id) {
        error_out = "Client returned an unexpected roots/list response id";
        return std::nullopt;
    }

    simdjson::ondemand::object error;
    if (root["error"].get_object().get(error) == simdjson::SUCCESS) {
        std::string_view message;
        if (error["message"].get_string().get(message) == simdjson::SUCCESS && !message.empty()) {
            error_out = std::string(message);
        } else {
            error_out = "Client rejected roots/list";
        }
        return std::nullopt;
    }

    simdjson::ondemand::object result;
    if (root["result"].get_object().get(result) != simdjson::SUCCESS) {
        error_out = "Client roots/list response was missing 'result'";
        return std::nullopt;
    }

    simdjson::ondemand::array roots;
    if (result["roots"].get_array().get(roots) != simdjson::SUCCESS) {
        error_out = "Client roots/list response was missing 'roots'";
        return std::nullopt;
    }

    core::workspace::WorkspaceSnapshot snapshot;
    snapshot.enforce = true;
    snapshot.version = workspace_version;

    bool first_root = true;
    for (auto root_item : roots) {
        simdjson::ondemand::object root_object;
        if (root_item.get_object().get(root_object) != simdjson::SUCCESS) {
            error_out = "Client roots/list entry was not an object";
            return std::nullopt;
        }

        std::string_view uri_value;
        if (root_object["uri"].get_string().get(uri_value) != simdjson::SUCCESS) {
            error_out = "Client roots/list entry was missing 'uri'";
            return std::nullopt;
        }

        std::string uri_error;
        auto parsed_path = parse_local_file_uri(uri_value, uri_error);
        if (!parsed_path.has_value()) {
            error_out = std::format("Invalid root URI '{}': {}",
                                    std::string(uri_value),
                                    uri_error);
            return std::nullopt;
        }

        if (first_root) {
            snapshot.primary = *parsed_path;
            first_root = false;
        } else {
            snapshot.additional.push_back(*parsed_path);
        }
    }

    return snapshot;
}

} // namespace core::mcp
