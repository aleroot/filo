#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <string>
#include <string_view>
#include <vector>

namespace core::agent {

// ---------------------------------------------------------------------------
// CommandSafetyClass — result of classifying a shell command.
// ---------------------------------------------------------------------------
enum class CommandSafetyClass {
    Safe,      // Pure read operations; no filesystem, network, or process changes.
    Dangerous, // Can mutate system state.
    Unknown,   // Unrecognised command — treated conservatively as Dangerous.
};

// ---------------------------------------------------------------------------
// CommandSafetyPolicy — classifies shell commands without executing them.
//
// Design principles:
//   • Conservative: Unknown → Dangerous (err on side of asking).
//   • Compositional: a chain is Safe only if ALL sub-commands are Safe.
//   • Git-aware: analyses git subcommands individually.
//   • Injection-aware: pipe-to-shell patterns are always Dangerous.
//
// Usage:
//   auto cls = CommandSafetyPolicy::classify("git log --oneline");
//   bool ask = CommandSafetyPolicy::command_needs_permission("rm -rf /tmp");
// ---------------------------------------------------------------------------
class CommandSafetyPolicy {
public:
    // Classify a full shell command string (may contain pipes, &&, ||, ;).
    [[nodiscard]] static CommandSafetyClass classify(std::string_view cmd) noexcept;

    // Returns true when user permission should be requested.
    [[nodiscard]] static bool command_needs_permission(std::string_view shell_cmd) noexcept {
        return classify(shell_cmd) != CommandSafetyClass::Safe;
    }

    // Extract the shell command string from a run_terminal_command JSON arg blob.
    // e.g. {"command":"ls -la","working_dir":"/tmp"}  →  "ls -la"
    [[nodiscard]] static std::string extract_shell_command(std::string_view tool_args_json) noexcept;

private:
    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    // Split a shell command at unquoted control operators (|, &&, ||, ;, \n).
    // Respects single-quoted, double-quoted, and backtick strings, plus $(…).
    [[nodiscard]] static std::vector<std::string> split_at_operators(std::string_view cmd) noexcept;

    // Extract all command-substitution bodies: $(body) and `body`.
    [[nodiscard]] static std::vector<std::string> extract_substitutions(std::string_view s) noexcept;

    // Classify a single pipeline segment (no &&/||/; remaining).
    [[nodiscard]] static CommandSafetyClass classify_segment(std::string_view segment) noexcept;

    // Classify once we have the bare command name and the rest of the args.
    [[nodiscard]] static CommandSafetyClass classify_command_name(
        std::string_view name, std::string_view rest) noexcept;

    // git-specific subcommand analysis.
    [[nodiscard]] static CommandSafetyClass classify_git(std::string_view args) noexcept;

    // -----------------------------------------------------------------------
    // String utilities
    // -----------------------------------------------------------------------

    [[nodiscard]] static std::string_view trim(std::string_view s) noexcept {
        while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
        while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))  s.remove_suffix(1);
        return s;
    }

    // Consume and return the next whitespace-delimited token from `rest`.
    // Note: not [[nodiscard]] — often called just to advance `rest`.
    static std::string_view next_token(std::string_view& rest) noexcept {
        // skip leading whitespace
        while (!rest.empty() && std::isspace(static_cast<unsigned char>(rest.front())))
            rest.remove_prefix(1);
        if (rest.empty()) return {};
        const auto start = rest.data();
        while (!rest.empty() && !std::isspace(static_cast<unsigned char>(rest.front())))
            rest.remove_prefix(1);
        return {start, static_cast<std::size_t>(rest.data() - start)};
    }

    // True if the token looks like a shell env-var assignment: FOO=bar or FOO=
    [[nodiscard]] static bool is_env_assignment(std::string_view tok) noexcept {
        if (tok.empty()) return false;
        if (!std::isalpha(static_cast<unsigned char>(tok[0])) && tok[0] != '_') return false;
        for (std::size_t i = 0; i < tok.size(); ++i) {
            const char c = tok[i];
            if (c == '=') return true;
            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') return false;
        }
        return false;
    }

    // Commands that are transparent wrappers (the real command follows).
    [[nodiscard]] static bool is_transparent_wrapper(std::string_view name) noexcept {
        static constexpr std::array<std::string_view, 8> wrappers = {
            "nohup", "time", "timeout", "nice", "ionice", "strace", "ltrace", "command",
        };
        return std::any_of(wrappers.begin(), wrappers.end(),
                           [name](std::string_view w) { return name == w; });
    }

    // Commands where the next argument is a numeric value to skip (e.g. "timeout 5 cmd").
    [[nodiscard]] static bool wrapper_takes_value_arg(std::string_view name) noexcept {
        return name == "timeout" || name == "nice" || name == "ionice";
    }
};

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

inline CommandSafetyClass CommandSafetyPolicy::classify(std::string_view cmd) noexcept {
    cmd = trim(cmd);
    if (cmd.empty()) return CommandSafetyClass::Safe;

    auto segments = split_at_operators(cmd);

    // Empty split → treat as single segment.
    if (segments.empty()) segments.emplace_back(cmd);

    CommandSafetyClass result = CommandSafetyClass::Safe;
    for (const auto& seg : segments) {
        const auto cls = classify_segment(trim(seg));
        if (cls == CommandSafetyClass::Dangerous) return CommandSafetyClass::Dangerous;
        if (cls == CommandSafetyClass::Unknown)   result = CommandSafetyClass::Unknown;
    }
    return result;
}

inline std::string CommandSafetyPolicy::extract_shell_command(std::string_view json) noexcept {
    // Find: "command" : "value"  (handles minor whitespace variations)
    const auto key = json.find("\"command\"");
    if (key == std::string_view::npos) return {};
    std::size_t pos = key + 9;  // skip past "command"
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
    if (pos >= json.size() || json[pos] != ':') return {};
    ++pos;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
    if (pos >= json.size() || json[pos] != '"') return {};
    ++pos;  // skip opening quote

    std::string result;
    while (pos < json.size()) {
        const char c = json[pos++];
        if (c == '\\' && pos < json.size()) {
            const char esc = json[pos++];
            switch (esc) {
                case '"':  result += '"';  break;
                case '\\': result += '\\'; break;
                case 'n':  result += '\n'; break;
                case 't':  result += '\t'; break;
                default:   result += esc;  break;
            }
        } else if (c == '"') {
            break;  // end of value
        } else {
            result += c;
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// split_at_operators — splits at |, ||, &&, ;, \n but not inside quotes.
// ---------------------------------------------------------------------------
inline std::vector<std::string> CommandSafetyPolicy::split_at_operators(std::string_view cmd) noexcept {
    std::vector<std::string> segments;
    std::string current;

    bool in_single = false;
    bool in_double = false;
    bool in_backtick = false;
    int  paren_depth = 0;

    const auto flush = [&] {
        if (!current.empty()) {
            segments.push_back(std::move(current));
            current.clear();
        }
    };

    for (std::size_t i = 0; i < cmd.size(); ++i) {
        const char c = cmd[i];

        // --- Inside single quotes -------------------------------------------
        if (in_single) {
            if (c == '\'') in_single = false;
            else current += c;
            continue;
        }

        // --- Inside backtick substitution -----------------------------------
        if (in_backtick) {
            if (c == '`') in_backtick = false;
            else current += c;
            continue;
        }

        // --- Inside double quotes -------------------------------------------
        if (in_double) {
            if (c == '\\' && i + 1 < cmd.size()) {
                current += cmd[++i];
            } else if (c == '"') {
                in_double = false;
            } else {
                current += c;
            }
            continue;
        }

        // --- Normal state ---------------------------------------------------
        if (c == '\'') { in_single   = true; continue; }
        if (c == '`')  { in_backtick = true; current += c; continue; }  // keep body for substitution extraction
        if (c == '"')  { in_double   = true; continue; }
        if (c == '(')  { ++paren_depth; current += c; continue; }
        if (c == ')') {
            if (paren_depth > 0) { --paren_depth; current += c; }
            continue;
        }

        // Don't split inside $(…)
        if (paren_depth > 0) { current += c; continue; }

        // Two-char operators
        if (i + 1 < cmd.size()) {
            if (c == '&' && cmd[i + 1] == '&') { flush(); ++i; continue; }
            if (c == '|' && cmd[i + 1] == '|') { flush(); ++i; continue; }
        }

        // One-char operators
        if (c == '|' || c == ';' || c == '\n') { flush(); continue; }

        // Background operator — split but don't classify the & itself.
        if (c == '&') { flush(); continue; }

        current += c;
    }
    flush();
    return segments;
}

// ---------------------------------------------------------------------------
// extract_substitutions — returns bodies of $(...) and `...` expressions.
// ---------------------------------------------------------------------------
inline std::vector<std::string> CommandSafetyPolicy::extract_substitutions(std::string_view s) noexcept {
    std::vector<std::string> bodies;

    for (std::size_t i = 0; i < s.size(); ++i) {
        // $(…) substitution
        if (i + 1 < s.size() && s[i] == '$' && s[i + 1] == '(') {
            i += 2;
            std::string body;
            int depth = 1;
            while (i < s.size() && depth > 0) {
                if (s[i] == '(') ++depth;
                else if (s[i] == ')') { if (--depth == 0) break; }
                body += s[i++];
            }
            if (!body.empty()) bodies.push_back(std::move(body));
            continue;
        }
        // `…` substitution
        if (s[i] == '`') {
            ++i;
            std::string body;
            while (i < s.size() && s[i] != '`') body += s[i++];
            if (!body.empty()) bodies.push_back(std::move(body));
            continue;
        }
    }
    return bodies;
}

// ---------------------------------------------------------------------------
// classify_segment — classify one pipeline element.
// ---------------------------------------------------------------------------
inline CommandSafetyClass CommandSafetyPolicy::classify_segment(std::string_view segment) noexcept {
    segment = trim(segment);
    if (segment.empty()) return CommandSafetyClass::Safe;

    // Recursively classify any command substitutions first.
    for (const auto& sub : extract_substitutions(segment)) {
        const auto cls = classify(sub);
        if (cls != CommandSafetyClass::Safe) return cls;
    }

    std::string_view rest = segment;

    // Skip leading env-var assignments: FOO=bar command …
    while (true) {
        const auto* before = rest.data();
        const auto tok = [&] {
            std::string_view tmp = rest;
            return next_token(tmp);
        }();
        if (is_env_assignment(tok)) {
            next_token(rest);  // consume it
        } else {
            break;
        }
        if (rest.data() == before) break;  // no progress guard
    }

    // Strip leading output redirections (>file, >>file, 2>&1, etc.) before
    // the actual command token — rare but valid shell syntax.
    // We look for tokens that start with > or < and skip them + their target.
    // This is a best-effort scan; deep redirect parsing is not needed.
    while (true) {
        std::string_view peek = rest;
        auto tok = next_token(peek);
        if (tok.empty()) return CommandSafetyClass::Safe;
        // If first char is > or < it's a redirection descriptor
        if (tok[0] == '>' || tok[0] == '<' ||
            (tok.size() > 1 && std::isdigit(static_cast<unsigned char>(tok[0])) &&
             (tok[1] == '>' || tok[1] == '<'))) {
            rest = peek; // consume the redirection token
            // If the redirect token is just the operator (not attached to filename),
            // also consume the target filename.
            if (tok == ">" || tok == ">>" || tok == "<" || tok == "2>" || tok == "2>>") {
                next_token(rest);
            }
        } else {
            break;
        }
    }

    // Extract command name.
    std::string_view cmd_name = next_token(rest);
    if (cmd_name.empty()) return CommandSafetyClass::Safe;

    // Unwrap transparent wrappers: nohup, time, nice …
    while (is_transparent_wrapper(cmd_name)) {
        if (wrapper_takes_value_arg(cmd_name)) next_token(rest); // skip numeric arg
        cmd_name = next_token(rest);
        if (cmd_name.empty()) return CommandSafetyClass::Safe;
    }

    // sudo always escalates to Dangerous regardless of what follows.
    if (cmd_name == "sudo" || cmd_name == "doas") return CommandSafetyClass::Dangerous;

    return classify_command_name(cmd_name, rest);
}

// ---------------------------------------------------------------------------
// classify_command_name — the core safe/dangerous knowledge base.
// ---------------------------------------------------------------------------
inline CommandSafetyClass CommandSafetyPolicy::classify_command_name(
    std::string_view name, std::string_view rest) noexcept
{
    // -----------------------------------------------------------------------
    // Definite shells / interpreters — always Dangerous.
    // -----------------------------------------------------------------------
    static constexpr std::array<std::string_view, 12> shells = {
        "bash", "sh", "zsh", "fish", "dash", "ksh", "tcsh", "csh",
        "python", "python3", "node", "ruby",
    };
    if (std::any_of(shells.begin(), shells.end(), [name](auto s) { return name == s; }))
        return CommandSafetyClass::Dangerous;

    // eval / exec / source / dot
    if (name == "eval" || name == "exec" || name == "source" || name == ".") return CommandSafetyClass::Dangerous;

    // -----------------------------------------------------------------------
    // Definite state-changers — always Dangerous.
    // -----------------------------------------------------------------------
    static constexpr std::array<std::string_view, 44> dangerous = {
        "rm", "rmdir", "shred", "unlink",           // deletion
        "mv", "cp", "install", "rsync",             // copy/move (can overwrite)
        "mkdir", "mkfifo", "mknod", "touch",        // creation / metadata
        "chmod", "chown", "chgrp", "chattr",        // permissions
        "ln",                                        // symlinks
        "dd", "truncate", "fallocate",              // low-level writes
        "kill", "killall", "pkill", "xkill",        // process signals
        "shutdown", "reboot", "halt", "poweroff",   // power
        "mount", "umount", "fdisk", "mkfs",         // disk
        "curl", "wget", "http", "httpie", "aria2c", // network/download
        "scp", "sftp", "ftp",                       // remote transfer
        "ssh",                                       // remote shell
        "make", "ninja", "cmake",                   // build systems (produce artefacts)
    };
    if (std::any_of(dangerous.begin(), dangerous.end(), [name](auto d) { return name == d; }))
        return CommandSafetyClass::Dangerous;

    // Package managers — always Dangerous.
    static constexpr std::array<std::string_view, 14> pkg_mgrs = {
        "apt", "apt-get", "dpkg", "rpm", "yum", "dnf", "pacman", "zypper",
        "brew", "snap", "flatpak", "npm", "yarn", "pnpm",
    };
    if (std::any_of(pkg_mgrs.begin(), pkg_mgrs.end(), [name](auto p) { return name == p; }))
        return CommandSafetyClass::Dangerous;

    // Language package managers / build tools — always Dangerous.
    static constexpr std::array<std::string_view, 7> lang_tools = {
        "pip", "pip3", "cargo", "go", "gem", "poetry", "conda",
    };
    if (std::any_of(lang_tools.begin(), lang_tools.end(), [name](auto t) { return name == t; }))
        return CommandSafetyClass::Dangerous;

    // Compilers — produce artefacts.
    static constexpr std::array<std::string_view, 5> compilers = {
        "gcc", "g++", "clang", "clang++", "javac",
    };
    if (std::any_of(compilers.begin(), compilers.end(), [name](auto c) { return name == c; }))
        return CommandSafetyClass::Dangerous;

    // Container / VM tools.
    static constexpr std::array<std::string_view, 4> containers = {
        "docker", "podman", "kubectl", "helm",
    };
    if (std::any_of(containers.begin(), containers.end(), [name](auto c) { return name == c; }))
        return CommandSafetyClass::Dangerous;

    // xargs — executes arbitrary commands.
    if (name == "xargs" || name == "parallel") return CommandSafetyClass::Dangerous;

    // tee — writes to files.
    if (name == "tee") return CommandSafetyClass::Dangerous;

    // -----------------------------------------------------------------------
    // Special: git — analyse subcommand.
    // -----------------------------------------------------------------------
    if (name == "git") return classify_git(rest);

    // -----------------------------------------------------------------------
    // Special: find — safe unless it runs executables (-exec, -delete, -ok).
    // -----------------------------------------------------------------------
    if (name == "find") {
        if (rest.find("-exec")     != std::string_view::npos) return CommandSafetyClass::Dangerous;
        if (rest.find("-execdir")  != std::string_view::npos) return CommandSafetyClass::Dangerous;
        if (rest.find("-ok")       != std::string_view::npos) return CommandSafetyClass::Dangerous;
        if (rest.find("-delete")   != std::string_view::npos) return CommandSafetyClass::Dangerous;
        return CommandSafetyClass::Safe;
    }

    // -----------------------------------------------------------------------
    // Special: sed — safe unless in-place edit flag is present.
    // -----------------------------------------------------------------------
    if (name == "sed") {
        // -i[suffix] means in-place edit
        if (rest.find("-i") != std::string_view::npos) return CommandSafetyClass::Dangerous;
        return CommandSafetyClass::Safe;
    }

    // -----------------------------------------------------------------------
    // Definite read-only commands — always Safe.
    // -----------------------------------------------------------------------
    static constexpr std::array<std::string_view, 73> safe_cmds = {
        // File inspection
        "cat", "head", "tail", "less", "more", "page",
        // Listing / navigation (read-only)
        "ls", "ll", "la", "dir", "pwd", "tree",
        // Search
        "grep", "egrep", "fgrep", "rg", "ag", "ack",
        // Text processing (stdout-only)
        "echo", "printf", "awk", "cut", "tr", "sort", "uniq", "column",
        "expand", "fold", "fmt",
        // File metadata
        "file", "stat", "wc", "du", "df", "free",
        "basename", "dirname", "readlink", "realpath",
        "md5sum", "sha1sum", "sha256sum", "sha512sum",
        "strings", "hexdump", "xxd", "od",
        // Process inspection
        "ps", "pgrep", "top", "htop", "uptime",
        // System info
        "uname", "hostname", "id", "whoami", "w", "who", "date", "cal",
        "env", "printenv",
        // Lookup
        "which", "type", "whereis", "command",
        // Diff (reading)
        "diff", "diff3", "colordiff",
        // JSON / data processing
        "jq",
        // Misc safe
        "true", "false", "test",
    };
    if (std::any_of(safe_cmds.begin(), safe_cmds.end(), [name](auto s) { return name == s; }))
        return CommandSafetyClass::Safe;

    // -----------------------------------------------------------------------
    // Fallback — unrecognised command, ask.
    // -----------------------------------------------------------------------
    return CommandSafetyClass::Unknown;
}

// ---------------------------------------------------------------------------
// classify_git — read-only vs state-changing git subcommands.
// ---------------------------------------------------------------------------
inline CommandSafetyClass CommandSafetyPolicy::classify_git(std::string_view args) noexcept {
    std::string_view rest = args;

    // Skip git global flags: --no-pager, -C /path, -c key=val, --git-dir=…, etc.
    while (true) {
        std::string_view peek = rest;
        const auto tok = next_token(peek);
        if (tok.empty()) break;
        // Flags that take a value argument
        if (tok == "-C" || tok == "-c" || tok == "--git-dir" ||
            tok == "--work-tree" || tok == "--namespace") {
            next_token(peek);  // skip value
            rest = peek;
        } else if (!tok.empty() && tok[0] == '-') {
            rest = peek;  // skip standalone flag
        } else {
            break;  // found the subcommand
        }
    }

    const auto sub = next_token(rest);
    if (sub.empty()) return CommandSafetyClass::Safe;  // bare "git" — no-op

    // Read-only subcommands.
    static constexpr std::array<std::string_view, 18> safe_subs = {
        "status", "log", "diff", "show", "shortlog",
        "branch",    // listing; -d/-D/-m would be dangerous but we check below
        "ls-files", "ls-tree", "ls-remote",
        "blame", "annotate",
        "describe", "rev-parse", "rev-list", "cat-file",
        "remote",    // listing; -v / --verbose is safe; "add"/"remove" is not
        "tag",       // listing; creating/deleting is not
        "config",    // read-only without --global/--local/--system + value
    };

    // Destructive subcommands.
    static constexpr std::array<std::string_view, 18> dangerous_subs = {
        "add", "commit", "push", "pull", "fetch",
        "merge", "rebase", "reset", "restore", "checkout", "switch",
        "clean", "rm", "mv",
        "cherry-pick", "revert", "bisect",
        "submodule",
    };

    if (std::any_of(dangerous_subs.begin(), dangerous_subs.end(), [sub](auto d) { return sub == d; }))
        return CommandSafetyClass::Dangerous;

    // stash: "stash list" / "stash show" are safe; everything else is not.
    if (sub == "stash") {
        const auto stash_sub = next_token(rest);
        if (stash_sub == "list" || stash_sub == "show") return CommandSafetyClass::Safe;
        if (stash_sub.empty()) return CommandSafetyClass::Dangerous; // bare "git stash" = git stash push
        return CommandSafetyClass::Dangerous;
    }

    // branch: safe when listing (-a, --list, -v, etc.), dangerous when deleting (-d/-D) or renaming (-m).
    if (sub == "branch") {
        const auto bflag = [&] {
            std::string_view tmp = rest;
            return next_token(tmp);
        }();
        if (bflag == "-d" || bflag == "-D" || bflag == "--delete" ||
            bflag == "-m" || bflag == "-M" || bflag == "--move" ||
            bflag == "-c" || bflag == "-C" || bflag == "--copy") return CommandSafetyClass::Dangerous;
        return CommandSafetyClass::Safe;
    }

    // remote: "remote -v" / "remote show" are safe; "remote add/remove" are not.
    if (sub == "remote") {
        const auto rflag = [&] {
            std::string_view tmp = rest;
            return next_token(tmp);
        }();
        if (rflag == "add" || rflag == "remove" || rflag == "rename" ||
            rflag == "set-url" || rflag == "set-head") return CommandSafetyClass::Dangerous;
        return CommandSafetyClass::Safe;
    }

    // tag: safe when listing, dangerous when creating/deleting.
    if (sub == "tag") {
        const auto tflag = [&] {
            std::string_view tmp = rest;
            return next_token(tmp);
        }();
        if (tflag == "-d" || tflag == "--delete" ||
            tflag == "-a" || tflag == "-s" || tflag == "-u") return CommandSafetyClass::Dangerous;
        // "git tag" with just a name creates a tag
        if (!tflag.empty() && tflag[0] != '-') return CommandSafetyClass::Dangerous;
        return CommandSafetyClass::Safe; // "git tag" / "git tag -l" = listing
    }

    // config: safe without write flags
    if (sub == "config") {
        const auto cflag = [&] {
            std::string_view tmp = rest;
            return next_token(tmp);
        }();
        if (cflag == "--global" || cflag == "--system" || cflag == "--local" ||
            cflag == "--add" || cflag == "--unset" || cflag == "--unset-all" ||
            cflag == "--replace-all" || cflag == "--rename-section" || cflag == "--remove-section")
            return CommandSafetyClass::Dangerous;
        return CommandSafetyClass::Safe;
    }

    if (std::any_of(safe_subs.begin(), safe_subs.end(), [sub](auto s) { return sub == s; }))
        return CommandSafetyClass::Safe;

    return CommandSafetyClass::Unknown;
}

} // namespace core::agent
