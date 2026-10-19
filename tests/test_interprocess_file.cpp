#include <catch2/catch_test_macros.hpp>

#include "core/utils/InterprocessFile.hpp"

#include <filesystem>
#include <fstream>
#include <sys/wait.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

struct TempDir {
    std::filesystem::path path = std::filesystem::temp_directory_path()
        / ("filo_interprocess_test_" + std::to_string(::getpid()));

    TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
        std::filesystem::create_directories(path);
    }

    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

} // namespace

TEST_CASE("InterprocessFileLock grants one owner and supports handoff", "[concurrency][file]") {
    TempDir temp;
    const auto lock_path = temp.path / "owner.lock";

    std::string error;
    auto first = core::utils::InterprocessFileLock::try_acquire(lock_path, &error);
    REQUIRE(first.has_value());
    CHECK(error.empty());

    auto follower = core::utils::InterprocessFileLock::try_acquire(lock_path, &error);
    CHECK_FALSE(follower.has_value());
    CHECK(error.empty());

    first.reset();
    follower = core::utils::InterprocessFileLock::try_acquire(lock_path, &error);
    CHECK(follower.has_value());
    CHECK(error.empty());
}

TEST_CASE("InterprocessFileLock excludes a separate process", "[concurrency][file]") {
    TempDir temp;
    const auto lock_path = temp.path / "process-owner.lock";
    int parent_to_child[2];
    int child_to_parent[2];
    REQUIRE(::pipe(parent_to_child) == 0);
    REQUIRE(::pipe(child_to_parent) == 0);

    const pid_t child = ::fork();
    REQUIRE(child >= 0);
    if (child == 0) {
        (void)::close(parent_to_child[1]);
        (void)::close(child_to_parent[0]);
        char signal = 0;
        if (::read(parent_to_child[0], &signal, 1) != 1) _exit(10);

        std::string error;
        auto blocked = core::utils::InterprocessFileLock::try_acquire(lock_path, &error);
        const char blocked_result = (!blocked && error.empty()) ? '1' : '0';
        if (::write(child_to_parent[1], &blocked_result, 1) != 1) _exit(11);

        if (::read(parent_to_child[0], &signal, 1) != 1) _exit(12);
        auto acquired = core::utils::InterprocessFileLock::try_acquire(lock_path, &error);
        _exit(acquired ? 0 : 13);
    }

    (void)::close(parent_to_child[0]);
    (void)::close(child_to_parent[1]);
    std::string error;
    auto owner = core::utils::InterprocessFileLock::try_acquire(lock_path, &error);
    REQUIRE(owner.has_value());

    const char signal = 'x';
    REQUIRE(::write(parent_to_child[1], &signal, 1) == 1);
    char blocked_result = 0;
    REQUIRE(::read(child_to_parent[0], &blocked_result, 1) == 1);
    CHECK(blocked_result == '1');

    owner.reset();
    REQUIRE(::write(parent_to_child[1], &signal, 1) == 1);
    int status = 0;
    REQUIRE(::waitpid(child, &status, 0) == child);
    CHECK(WIFEXITED(status));
    CHECK(WEXITSTATUS(status) == 0);

    (void)::close(parent_to_child[1]);
    (void)::close(child_to_parent[0]);
}

TEST_CASE("atomic_write_file replaces content without shared temp names", "[concurrency][file]") {
    TempDir temp;
    const auto target = temp.path / "state.json";
    std::string error;

    REQUIRE(core::utils::atomic_write_file(target, "{\"value\":1}\n", &error));
    REQUIRE(core::utils::atomic_write_file(target, "{\"value\":2}\n", &error));

    std::ifstream input(target, std::ios::binary);
    const std::string content{
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    CHECK(content == "{\"value\":2}\n");

    for (const auto& entry : std::filesystem::directory_iterator(temp.path)) {
        CHECK_FALSE(entry.path().filename().string().contains(".tmp."));
    }
}

TEST_CASE("Bounded regular file reads distinguish absence, empty files and unsafe paths", "[file][recovery]") {
    TempDir temp;
    const auto path = temp.path / "source";
    auto missing = core::utils::read_regular_file(path, 4);
    REQUIRE(missing);
    CHECK_FALSE(missing->permissions);

    REQUIRE(core::utils::atomic_write_file(path, "", nullptr,
        std::filesystem::perms::owner_all));
    auto empty = core::utils::read_regular_file(path, 0);
    REQUIRE(empty);
    REQUIRE(empty->permissions);
    CHECK(*empty->permissions == std::filesystem::perms::owner_all);
    CHECK(empty->bytes.empty());

    const std::string binary("a\0bc", 4);
    REQUIRE(core::utils::atomic_write_file(path, binary));
    auto contents = core::utils::read_regular_file(path, binary.size());
    REQUIRE(contents);
    CHECK(contents->bytes == binary);
    CHECK_FALSE(core::utils::read_regular_file(path, binary.size() - 1));
    std::filesystem::create_symlink(path, temp.path / "link");
    CHECK_FALSE(core::utils::read_regular_file(temp.path / "link", 4));
    CHECK_FALSE(core::utils::read_regular_file(temp.path, 4));
    REQUIRE(::mkfifo((temp.path / "pipe").c_str(), 0600) == 0);
    CHECK_FALSE(core::utils::read_regular_file(temp.path / "pipe", 4));
}
