#include "EvidenceSelector.hpp"
#include "../../utils/StringUtils.hpp"
#include "../../utils/AsciiUtils.hpp"
#include <algorithm>
#include <format>

namespace core::tools::read {
namespace {
std::vector<std::string> keywords(std::string_view question) {
    std::vector<std::string> terms;
    std::string word;
    auto flush = [&] {
        if (word.size() >= 3 && terms.size() < 16 && std::ranges::find(terms, word) == terms.end()
            && word != "the" && word != "what" && word != "does" && word != "how"
            && word != "and" && word != "are" && word != "this" && word != "that") terms.push_back(word);
        word.clear();
    };
    for (const unsigned char c : question) {
        if (core::utils::ascii::is_alnum(c) || c == '_')
            word += core::utils::ascii::to_lower(static_cast<char>(c));
        else flush();
    }
    flush();
    return terms;
}
} // namespace
Evidence select_evidence(std::span<const Resource> sources, std::string_view question, std::size_t budget) {
    Evidence evidence;
    if (sources.empty()) return evidence;
    const auto terms = keywords(question);
    const auto share = budget / sources.size();
    for (std::size_t i = 0; i < sources.size(); ++i) {
        const auto& source = sources[i];
        const auto entries = lines(source.text);
        std::vector<std::size_t> selected;
        std::vector<bool> included(entries.size(), false);
        std::string header = std::format("Source {} ({}, path={}):\n", i + 1, source.kind, bounded_prefix(source.uri, 1024));
        std::size_t used = header.size() + 64;
        auto add = [&](std::size_t line) {
            if (line >= entries.size() || included[line]) return;
            const auto bytes = entries[line].size() + 24; // room for line number and delimiter
            if (used + bytes > share) return;
            used += bytes;
            selected.push_back(line + 1);
            included[line] = true;
        };
        // A small head gives context. Rank a bounded candidate set for the
        // remainder, so a question can reach declarations late in a large file.
        for (std::size_t line = 0; line < std::min<std::size_t>(8, entries.size()); ++line) add(line);
        struct Candidate { std::size_t line; int score; };
        std::vector<Candidate> best;
        auto better = [](const Candidate& a, const Candidate& b) {
            return a.score != b.score ? a.score > b.score : a.line < b.line;
        };
        // Lowercase each candidate line once rather than once per term: this
        // scan covers up to 2 MiB of source against up to 16 terms.
        std::string lowered;
        for (std::size_t line = 8; !terms.empty() && line < entries.size(); ++line) {
            lowered.assign(entries[line]);
            std::ranges::transform(lowered, lowered.begin(), [](unsigned char c) {
                return core::utils::ascii::to_lower(static_cast<char>(c));
            });
            int score = 0;
            for (const auto& term : terms)
                if (lowered.find(term) != std::string::npos) ++score;
            if (!score) continue;
            if (best.size() == 24 && !better({line, score}, best.back())) continue;
            best.push_back({line, score});
            std::ranges::sort(best, better);
            if (best.size() > 24) best.pop_back();
        }
        for (const auto& match : best) {
            add(match.line);
            if (match.line) add(match.line - 1);
            add(match.line + 1);
        }
        // Small sources fit in full. Larger ones fill unused capacity in order.
        for (std::size_t line = 8; line < entries.size() && used + 24 < share; ++line) add(line);
        std::ranges::sort(selected);
        evidence.text += header;
        for (auto line : selected) evidence.text += std::format("{}: {}\n", line, entries[line - 1]);
        evidence.text += "[End source; gaps are omitted and have not been analyzed]\n";
        evidence.partial |= source.truncated || selected.size() != entries.size();
        evidence.supplied_lines.push_back(std::move(selected));
    }
    return evidence;
}
} // namespace core::tools::read
