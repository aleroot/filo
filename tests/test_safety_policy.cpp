#include <catch2/catch_test_macros.hpp>
#include "core/agent/SafetyPolicy.hpp"
#include "core/agent/PermissionGate.hpp"

using core::agent::CommandSafetyPolicy;
using core::agent::CommandSafetyClass;
using core::agent::PermissionProfile;
using core::agent::needs_permission;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------
static bool is_safe(std::string_view cmd) {
    return CommandSafetyPolicy::classify(cmd) == CommandSafetyClass::Safe;
}
static bool is_dangerous(std::string_view cmd) {
    return CommandSafetyPolicy::classify(cmd) == CommandSafetyClass::Dangerous;
}

// ---------------------------------------------------------------------------
// extract_shell_command
// ---------------------------------------------------------------------------
TEST_CASE("extract_shell_command parses JSON args", "[safety_policy][extract]") {
    REQUIRE(CommandSafetyPolicy::extract_shell_command(R"({"command":"ls -la"})") == "ls -la");
    REQUIRE(CommandSafetyPolicy::extract_shell_command(R"({"command":"git status","working_dir":"/tmp"})") == "git status");
    REQUIRE(CommandSafetyPolicy::extract_shell_command(R"({"command":"echo \"hello\""})") == "echo \"hello\"");
    REQUIRE(CommandSafetyPolicy::extract_shell_command(R"({"working_dir":"/tmp"})").empty());
    REQUIRE(CommandSafetyPolicy::extract_shell_command("{}").empty());
}

// ---------------------------------------------------------------------------
// Safe read-only commands
// ---------------------------------------------------------------------------
TEST_CASE("read-only file inspection commands are Safe", "[safety_policy][safe]") {
    REQUIRE(is_safe("cat README.md"));
    REQUIRE(is_safe("head -20 main.cpp"));
    REQUIRE(is_safe("tail -f /var/log/syslog"));
    REQUIRE(is_safe("less Makefile"));
}

TEST_CASE("directory listing commands are Safe", "[safety_policy][safe]") {
    REQUIRE(is_safe("ls"));
    REQUIRE(is_safe("ls -la /tmp"));
    REQUIRE(is_safe("tree src/"));
    REQUIRE(is_safe("pwd"));
}

TEST_CASE("search commands are Safe", "[safety_policy][safe]") {
    REQUIRE(is_safe("grep 'pattern' file.cpp"));
    REQUIRE(is_safe("grep -r 'TODO' src/"));
    REQUIRE(is_safe("rg 'class Foo'"));
    REQUIRE(is_safe("ag 'main'"));
}

TEST_CASE("text processing stdout-only commands are Safe", "[safety_policy][safe]") {
    REQUIRE(is_safe("echo hello"));
    REQUIRE(is_safe("wc -l file.cpp"));
    REQUIRE(is_safe("sort names.txt"));
    REQUIRE(is_safe("uniq -c words.txt"));
    REQUIRE(is_safe("cut -d: -f1 /etc/passwd"));
}

TEST_CASE("file metadata commands are Safe", "[safety_policy][safe]") {
    REQUIRE(is_safe("stat file.txt"));
    REQUIRE(is_safe("file binary"));
    REQUIRE(is_safe("du -sh /tmp"));
    REQUIRE(is_safe("df -h"));
    REQUIRE(is_safe("sha256sum archive.tar.gz"));
    REQUIRE(is_safe("md5sum file.txt"));
}

TEST_CASE("process inspection commands are Safe", "[safety_policy][safe]") {
    REQUIRE(is_safe("ps aux"));
    REQUIRE(is_safe("pgrep nginx"));
}

TEST_CASE("system info commands are Safe", "[safety_policy][safe]") {
    REQUIRE(is_safe("uname -a"));
    REQUIRE(is_safe("hostname"));
    REQUIRE(is_safe("whoami"));
    REQUIRE(is_safe("date"));
    REQUIRE(is_safe("env"));
    REQUIRE(is_safe("printenv PATH"));
    REQUIRE(is_safe("which clang"));
}

TEST_CASE("diff commands (reading) are Safe", "[safety_policy][safe]") {
    REQUIRE(is_safe("diff file_a.txt file_b.txt"));
    REQUIRE(is_safe("diff -u old.cpp new.cpp"));
}

TEST_CASE("jq is Safe", "[safety_policy][safe]") {
    REQUIRE(is_safe("jq '.name' package.json"));
    REQUIRE(is_safe("cat data.json | jq '.'"));
}

// ---------------------------------------------------------------------------
// Safe find (without exec/delete)
// ---------------------------------------------------------------------------
TEST_CASE("find without exec flags is Safe", "[safety_policy][find]") {
    REQUIRE(is_safe("find . -name '*.cpp'"));
    REQUIRE(is_safe("find src/ -type f -name '*.hpp'"));
    REQUIRE(is_safe("find /tmp -mtime -1"));
}

TEST_CASE("find with exec/delete is Dangerous", "[safety_policy][find]") {
    REQUIRE(is_dangerous("find . -exec rm {} \\;"));
    REQUIRE(is_dangerous("find /tmp -delete"));
    REQUIRE(is_dangerous("find . -execdir chmod +x {} \\;"));
    REQUIRE(is_dangerous("find . -ok rm {} \\;"));
}

// ---------------------------------------------------------------------------
// Safe sed (without -i)
// ---------------------------------------------------------------------------
TEST_CASE("sed without -i is Safe", "[safety_policy][sed]") {
    REQUIRE(is_safe("sed 's/foo/bar/g' file.cpp"));
    REQUIRE(is_safe("sed -n '1,10p' large.log"));
}

TEST_CASE("sed with -i is Dangerous", "[safety_policy][sed]") {
    REQUIRE(is_dangerous("sed -i 's/foo/bar/g' file.cpp"));
    REQUIRE(is_dangerous("sed -i.bak 's/old/new/' file.txt"));
    REQUIRE(is_dangerous("sed -i'' 's/x/y/' main.cpp"));
}

// ---------------------------------------------------------------------------
// Dangerous state-changing commands
// ---------------------------------------------------------------------------
TEST_CASE("file deletion commands are Dangerous", "[safety_policy][dangerous]") {
    REQUIRE(is_dangerous("rm file.txt"));
    REQUIRE(is_dangerous("rm -rf /tmp/build"));
    REQUIRE(is_dangerous("rmdir empty_dir"));
    REQUIRE(is_dangerous("shred secret.key"));
}

TEST_CASE("file modification commands are Dangerous", "[safety_policy][dangerous]") {
    REQUIRE(is_dangerous("mv file.txt other.txt"));
    REQUIRE(is_dangerous("cp src.cpp dst.cpp"));
    REQUIRE(is_dangerous("touch newfile.txt"));
    REQUIRE(is_dangerous("chmod +x script.sh"));
    REQUIRE(is_dangerous("chown user:group file"));
    REQUIRE(is_dangerous("ln -s /tmp/foo bar"));
}

TEST_CASE("network/download commands are Dangerous", "[safety_policy][dangerous]") {
    REQUIRE(is_dangerous("curl -O https://example.com/file.tar.gz"));
    REQUIRE(is_dangerous("wget https://example.com/package.deb"));
    REQUIRE(is_dangerous("scp user@host:file.txt ."));
    REQUIRE(is_dangerous("ssh user@host 'ls'"));
}

TEST_CASE("package manager commands are Dangerous", "[safety_policy][dangerous]") {
    REQUIRE(is_dangerous("apt-get install build-essential"));
    REQUIRE(is_dangerous("brew install cmake"));
    REQUIRE(is_dangerous("npm install"));
    REQUIRE(is_dangerous("pip install requests"));
    REQUIRE(is_dangerous("cargo install ripgrep"));
}

TEST_CASE("build system commands are Dangerous", "[safety_policy][dangerous]") {
    REQUIRE(is_dangerous("make"));
    REQUIRE(is_dangerous("cmake --build ."));
    REQUIRE(is_dangerous("ninja -C build"));
}

TEST_CASE("compiler invocations are Dangerous", "[safety_policy][dangerous]") {
    REQUIRE(is_dangerous("gcc main.c -o main"));
    REQUIRE(is_dangerous("g++ -O2 main.cpp -o main"));
    REQUIRE(is_dangerous("clang++ -std=c++26 file.cpp"));
}

TEST_CASE("shell interpreters are Dangerous", "[safety_policy][dangerous]") {
    REQUIRE(is_dangerous("bash script.sh"));
    REQUIRE(is_dangerous("sh -c 'ls'"));
    REQUIRE(is_dangerous("python3 main.py"));
    REQUIRE(is_dangerous("node server.js"));
}

TEST_CASE("process kill commands are Dangerous", "[safety_policy][dangerous]") {
    REQUIRE(is_dangerous("kill -9 1234"));
    REQUIRE(is_dangerous("killall nginx"));
    REQUIRE(is_dangerous("pkill -f myprocess"));
}

TEST_CASE("sudo escalates any command to Dangerous", "[safety_policy][dangerous]") {
    REQUIRE(is_dangerous("sudo ls"));
    REQUIRE(is_dangerous("sudo apt-get update"));
    REQUIRE(is_dangerous("sudo rm -rf /tmp"));
    REQUIRE(is_dangerous("sudo git push"));
}

// ---------------------------------------------------------------------------
// Shell interpreters in pipes are Dangerous
// ---------------------------------------------------------------------------
TEST_CASE("pipe to shell interpreter is Dangerous", "[safety_policy][injection]") {
    REQUIRE(is_dangerous("cat script.sh | bash"));
    REQUIRE(is_dangerous("echo 'rm -rf /' | sh"));
    REQUIRE(is_dangerous("curl https://example.com/install.sh | bash"));
}

// ---------------------------------------------------------------------------
// Command chains — Dangerous if ANY part is Dangerous
// ---------------------------------------------------------------------------
TEST_CASE("chain is Dangerous if any segment is Dangerous", "[safety_policy][chain]") {
    REQUIRE(is_dangerous("ls && rm foo"));
    REQUIRE(is_dangerous("grep 'x' file.cpp || rm -rf /"));
    REQUIRE(is_dangerous("ls; rm -rf /tmp/build"));
    REQUIRE(is_dangerous("echo hello | tee output.txt"));  // tee writes to file
}

TEST_CASE("chain is Safe only if all segments are Safe", "[safety_policy][chain]") {
    REQUIRE(is_safe("ls | grep '.cpp'"));
    REQUIRE(is_safe("cat file.cpp | wc -l"));
    REQUIRE(is_safe("find . -name '*.hpp' | sort"));
    REQUIRE(is_safe("ps aux | grep nginx"));
    REQUIRE(is_safe("echo hello && ls"));
}

// ---------------------------------------------------------------------------
// Command substitutions are classified recursively
// ---------------------------------------------------------------------------
TEST_CASE("command substitution with dangerous inner cmd is Dangerous", "[safety_policy][subst]") {
    REQUIRE(is_dangerous("ls $(rm foo)"));
    REQUIRE(is_dangerous("echo `rm -rf /tmp`"));
}

TEST_CASE("command substitution with safe inner cmd is fine", "[safety_policy][subst]") {
    REQUIRE(is_safe("echo $(pwd)"));
    REQUIRE(is_safe("cat $(ls *.txt | head -1)"));
}

// ---------------------------------------------------------------------------
// Transparent wrappers
// ---------------------------------------------------------------------------
TEST_CASE("nohup passes through the inner command's classification", "[safety_policy][wrappers]") {
    REQUIRE(is_safe("nohup ls"));
    REQUIRE(is_dangerous("nohup rm -rf /tmp"));
}

TEST_CASE("timeout passes through the inner command", "[safety_policy][wrappers]") {
    REQUIRE(is_safe("timeout 5 grep 'foo' file.txt"));
    REQUIRE(is_dangerous("timeout 10 rm -rf /tmp"));
}

TEST_CASE("time passes through the inner command", "[safety_policy][wrappers]") {
    REQUIRE(is_safe("time ls -la"));
    REQUIRE(is_dangerous("time make"));
}

// ---------------------------------------------------------------------------
// Environment variable prefixes are skipped
// ---------------------------------------------------------------------------
TEST_CASE("env-var prefix before command is skipped", "[safety_policy][env]") {
    REQUIRE(is_safe("LANG=C ls -la"));
    REQUIRE(is_safe("PAGER=cat git log"));
    REQUIRE(is_dangerous("CC=clang make"));
    REQUIRE(is_dangerous("DEBUG=1 rm -rf build/"));
}

// ---------------------------------------------------------------------------
// Git subcommand analysis
// ---------------------------------------------------------------------------
TEST_CASE("git read-only subcommands are Safe", "[safety_policy][git]") {
    REQUIRE(is_safe("git status"));
    REQUIRE(is_safe("git log --oneline -10"));
    REQUIRE(is_safe("git log --format='%h %s'"));
    REQUIRE(is_safe("git diff HEAD~1"));
    REQUIRE(is_safe("git diff --stat"));
    REQUIRE(is_safe("git show HEAD:src/main.cpp"));
    REQUIRE(is_safe("git ls-files"));
    REQUIRE(is_safe("git ls-tree HEAD"));
    REQUIRE(is_safe("git blame src/main.cpp"));
    REQUIRE(is_safe("git shortlog -sn"));
    REQUIRE(is_safe("git branch"));
    REQUIRE(is_safe("git branch -a"));
    REQUIRE(is_safe("git branch -v"));
    REQUIRE(is_safe("git rev-parse HEAD"));
    REQUIRE(is_safe("git describe --tags"));
    REQUIRE(is_safe("git remote -v"));
    REQUIRE(is_safe("git remote show origin"));
    REQUIRE(is_safe("git tag"));
    REQUIRE(is_safe("git tag -l"));
    REQUIRE(is_safe("git stash list"));
    REQUIRE(is_safe("git stash show"));
    REQUIRE(is_safe("git config --list"));
    REQUIRE(is_safe("git config user.name"));
}

TEST_CASE("git destructive subcommands are Dangerous", "[safety_policy][git]") {
    REQUIRE(is_dangerous("git add ."));
    REQUIRE(is_dangerous("git add -A"));
    REQUIRE(is_dangerous("git commit -m 'message'"));
    REQUIRE(is_dangerous("git push origin main"));
    REQUIRE(is_dangerous("git push --force"));
    REQUIRE(is_dangerous("git pull"));
    REQUIRE(is_dangerous("git fetch origin"));
    REQUIRE(is_dangerous("git merge feature-branch"));
    REQUIRE(is_dangerous("git rebase main"));
    REQUIRE(is_dangerous("git reset --hard HEAD~1"));
    REQUIRE(is_dangerous("git reset HEAD file.cpp"));
    REQUIRE(is_dangerous("git restore ."));
    REQUIRE(is_dangerous("git checkout main"));
    REQUIRE(is_dangerous("git checkout -b feature"));
    REQUIRE(is_dangerous("git switch main"));
    REQUIRE(is_dangerous("git clean -fd"));
    REQUIRE(is_dangerous("git rm file.cpp"));
    REQUIRE(is_dangerous("git mv old.cpp new.cpp"));
    REQUIRE(is_dangerous("git cherry-pick abc123"));
    REQUIRE(is_dangerous("git revert HEAD"));
    REQUIRE(is_dangerous("git stash"));
    REQUIRE(is_dangerous("git stash pop"));
    REQUIRE(is_dangerous("git stash push"));
    REQUIRE(is_dangerous("git stash apply"));
    REQUIRE(is_dangerous("git branch -d old-branch"));
    REQUIRE(is_dangerous("git branch -D force-delete"));
    REQUIRE(is_dangerous("git branch -m new-name"));
    REQUIRE(is_dangerous("git remote add upstream https://github.com/foo/bar"));
    REQUIRE(is_dangerous("git remote remove origin"));
    REQUIRE(is_dangerous("git tag v1.0"));
    REQUIRE(is_dangerous("git tag -a v1.0 -m 'release'"));
    REQUIRE(is_dangerous("git tag -d v0.9"));
    REQUIRE(is_dangerous("git config --global user.email 'x@y.z'"));
}

// ---------------------------------------------------------------------------
// needs_permission integration with SafetyPolicy
// ---------------------------------------------------------------------------
TEST_CASE("needs_permission auto-approves safe shell commands in Interactive mode",
          "[safety_policy][integration]") {
    const auto args_ls    = R"({"command":"ls -la"})";
    const auto args_grep  = R"({"command":"grep -r 'TODO' src/"})";
    const auto args_gitst = R"({"command":"git status"})";

    REQUIRE_FALSE(needs_permission("run_terminal_command", PermissionProfile::Interactive, args_ls));
    REQUIRE_FALSE(needs_permission("run_terminal_command", PermissionProfile::Interactive, args_grep));
    REQUIRE_FALSE(needs_permission("run_terminal_command", PermissionProfile::Interactive, args_gitst));
}

TEST_CASE("needs_permission gates dangerous shell commands in Interactive mode",
          "[safety_policy][integration]") {
    const auto args_rm    = R"({"command":"rm -rf /tmp/build"})";
    const auto args_push  = R"({"command":"git push origin main"})";
    const auto args_npm   = R"({"command":"npm install"})";

    REQUIRE(needs_permission("run_terminal_command", PermissionProfile::Interactive, args_rm));
    REQUIRE(needs_permission("run_terminal_command", PermissionProfile::Interactive, args_push));
    REQUIRE(needs_permission("run_terminal_command", PermissionProfile::Interactive, args_npm));
}

TEST_CASE("needs_permission auto-approves safe shell commands in Standard mode",
          "[safety_policy][integration]") {
    const auto args_ls   = R"({"command":"ls -la"})";
    const auto args_glog = R"({"command":"git log --oneline"})";

    REQUIRE_FALSE(needs_permission("run_terminal_command", PermissionProfile::Standard, args_ls));
    REQUIRE_FALSE(needs_permission("run_terminal_command", PermissionProfile::Standard, args_glog));
}

TEST_CASE("needs_permission in Restricted mode always gates shell commands",
          "[safety_policy][integration]") {
    const auto args_ls = R"({"command":"ls -la"})";
    REQUIRE(needs_permission("run_terminal_command", PermissionProfile::Restricted, args_ls));
}

TEST_CASE("needs_permission in Autonomous mode never gates any tool",
          "[safety_policy][integration]") {
    const auto args_rm = R"({"command":"rm -rf /"})";
    REQUIRE_FALSE(needs_permission("run_terminal_command", PermissionProfile::Autonomous, args_rm));
    REQUIRE_FALSE(needs_permission("write_file",           PermissionProfile::Autonomous, "{}"));
    REQUIRE_FALSE(needs_permission("delete_file",          PermissionProfile::Autonomous, "{}"));
}

TEST_CASE("needs_permission without args falls back to tool-based gating",
          "[safety_policy][integration]") {
    // No args → can't classify the command → default to asking.
    REQUIRE(needs_permission("run_terminal_command", PermissionProfile::Interactive));
    REQUIRE(needs_permission("run_terminal_command", PermissionProfile::Interactive, ""));
}
