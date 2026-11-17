#include <catch2/catch_test_macros.hpp>

#include "core/persistence/AmendedFile.hpp"
#include "core/utils/InterprocessFile.hpp"

#include <atomic>
#include <filesystem>
#include <format>
#include <fstream>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
using core::persistence::AmendedFile;

namespace {

struct TempDir {
    fs::path path;
    TempDir() {
        static std::atomic<unsigned> counter{0};
        path = fs::temp_directory_path()
            / std::format("filo_amended_{}_{}_{}", ::getpid(), std::random_device{}(), counter.fetch_add(1));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

AmendedFile::Contents read_back(const fs::path& path) {
    auto contents = AmendedFile::read(path, 1 << 20);
    REQUIRE(contents);
    REQUIRE(*contents);
    return std::move(**contents);
}

std::string bytes_of(const fs::path& path) { return core::utils::read_file(path).value_or(""); }

} // namespace

TEST_CASE("An amended file reads back as its snapshot and records, in order", "[persistence]") {
    TempDir dir;
    const auto path = dir.path / "state.json";
    CHECK_FALSE(AmendedFile::read(path, 1024).value());  // Nothing saved yet.

    AmendedFile file(path);
    CHECK_FALSE(file.amend("before any snapshot"));
    REQUIRE(file.replace("snapshot"));
    CHECK_FALSE(fs::exists(AmendedFile::log_path(path)));
    REQUIRE(file.amend("first"));
    REQUIRE(file.amend("", AmendedFile::Durability::Deferred));  // Empty records are records.
    REQUIRE(file.amend("third"));
    REQUIRE(file.sync());
    CHECK(bytes_of(path) == "snapshot");

    const auto contents = read_back(path);
    CHECK(contents.snapshot == "snapshot");
    CHECK(contents.records == std::vector<std::string>{"first", "", "third"});

    SECTION("a reader's position lets another writer continue the log") {
        AmendedFile other(path);
        other.adopt(contents.position);
        REQUIRE(other.amend("fourth"));
        CHECK(read_back(path).records.back() == "fourth");
        // The first writer's view is stale now: it must not append blindly.
        CHECK_FALSE(file.amend("stale"));
    }

    SECTION("replacing the snapshot drops every record") {
        REQUIRE(file.replace("new snapshot"));
        CHECK_FALSE(fs::exists(AmendedFile::log_path(path)));
        const auto replaced = read_back(path);
        CHECK(replaced.snapshot == "new snapshot");
        CHECK(replaced.records.empty());
    }

    SECTION("a snapshot another writer replaced is not amended") {
        AmendedFile(path).replace("someone else's").value();
        CHECK_FALSE(file.amend("lost"));
        CHECK(read_back(path).records.empty());
    }
}

TEST_CASE("A torn or corrupt record ends what an amended file reads", "[persistence]") {
    TempDir dir;
    const auto path = dir.path / "state";
    AmendedFile file(path);
    REQUIRE(file.replace("snapshot"));
    REQUIRE(file.amend("kept"));
    REQUIRE(file.amend("damaged"));
    REQUIRE(file.amend("after the damage"));
    const auto log = AmendedFile::log_path(path);
    auto bytes = bytes_of(log);

    SECTION("torn") {
        std::ofstream(log, std::ios::binary | std::ios::trunc) << bytes.substr(0, bytes.size() - 3);
        const auto contents = read_back(path);
        CHECK(contents.records == std::vector<std::string>{"kept", "damaged"});
        // Nothing may follow a torn record: a writer must replace instead.
        CHECK_FALSE(contents.position.log_bytes);
        AmendedFile other(path);
        other.adopt(contents.position);
        CHECK_FALSE(other.amend("more"));
    }

    SECTION("corrupt") {
        bytes[bytes.find("damaged")] = 'D';
        std::ofstream(log, std::ios::binary | std::ios::trunc) << bytes;
        CHECK(read_back(path).records == std::vector<std::string>{"kept"});
    }
}

TEST_CASE("An amended file never applies a log written for another snapshot", "[persistence]") {
    TempDir dir;
    const auto path = dir.path / "state";
    AmendedFile file(path);
    REQUIRE(file.replace("one"));
    REQUIRE(file.amend("change to one"));
    const auto stale = bytes_of(AmendedFile::log_path(path));

    SECTION("a crash after the new snapshot, before the old log was removed") {
        REQUIRE(file.replace("two"));
        std::ofstream(AmendedFile::log_path(path), std::ios::binary) << stale;
        const auto contents = read_back(path);
        CHECK(contents.snapshot == "two");
        CHECK(contents.records.empty());
        CHECK_FALSE(contents.position.log_bytes);  // And it is not extended either.
    }

    SECTION("a new snapshot of the very bytes the log amends") {
        // The log would amend the new snapshot too, so replace() drops it
        // before writing: a crash in between still reads "one", unamended.
        REQUIRE(file.replace("one"));
        CHECK(read_back(path).records.empty());
    }
}

TEST_CASE("An amended file refuses records once its log outgrows the snapshot", "[persistence]") {
    TempDir dir;
    const auto path = dir.path / "state";
    AmendedFile file(path, /*min_log_bytes=*/64);
    REQUIRE(file.replace(std::string(100, 's')));
    int accepted = 0;
    while (file.amend(std::string(10, 'r'))) ++accepted;
    // A header and 18-byte frames fit in 100 bytes a few times, never more.
    CHECK(accepted > 0);
    CHECK(fs::file_size(AmendedFile::log_path(path)) <= 100);
    CHECK(read_back(path).records.size() == static_cast<std::size_t>(accepted));
}
