#include <catch2/catch_test_macros.hpp>

#include "core/changes/ObjectStore.hpp"
#include "core/utils/Crypto.hpp"
#include "core/utils/FileClone.hpp"
#include "core/utils/InterprocessFile.hpp"

#include <atomic>
#include <filesystem>
#include <format>
#include <fstream>
#include <random>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;
using core::changes::ObjectStore;

namespace {

struct TempDir {
    fs::path path;
    TempDir() {
        static std::atomic<unsigned> counter{0};
        path = fs::temp_directory_path()
            / std::format("filo_objects_{}_{}_{}", ::getpid(), std::random_device{}(), counter.fetch_add(1));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

void put(const fs::path& path, std::string_view bytes) {
    std::ofstream(path, std::ios::binary | std::ios::trunc) << bytes;
}

ino_t inode_of(const fs::path& path) {
    struct stat status{};
    REQUIRE(::stat(path.c_str(), &status) == 0);
    return status.st_ino;
}

} // namespace

TEST_CASE("An object store keeps each content once, under its hash", "[changes][objects]") {
    TempDir dir;
    const auto file = dir.path / "file.txt";
    put(file, "contents\n");
    const ObjectStore store(dir.path / "objects");

    const auto stored = store.store(file, 1024);
    REQUIRE(stored);
    CHECK(stored->image.hash == core::utils::crypto::sha256_hex("contents\n"));
    const auto object = store.directory() / stored->image.hash;
    CHECK(store.contains(stored->image.hash));
    CHECK((fs::status(object).permissions() & fs::perms::all) == (fs::perms::owner_read | fs::perms::owner_write));
    const auto inode = inode_of(object);

    // The same contents again, from another file: nothing is written.
    put(dir.path / "copy.txt", "contents\n");
    REQUIRE(store.store(dir.path / "copy.txt", 1024));
    CHECK(inode_of(object) == inode);

    // An object is its own: the file it came from may change.
    put(file, "edited\n");
    CHECK(store.read(stored->image.hash, 1024).value_or("") == "contents\n");

    CHECK(store.store(dir.path / "absent", 1024).value().image.hash.empty());
    CHECK_FALSE(store.store(file, 3));  // Over the limit: nothing stored.

    // Sweeping keeps what is referenced and anything not named like an object.
    put(store.directory() / "tmp.in-progress", "");
    const auto edited = store.store(file, 1024).value();
    store.sweep({stored->image.hash});
    CHECK(store.contains(stored->image.hash));
    CHECK_FALSE(store.contains(edited.image.hash));
    CHECK(fs::exists(store.directory() / "tmp.in-progress"));
}

TEST_CASE("An object store links what a peer store already holds", "[changes][objects]") {
    TempDir dir;
    const auto file = dir.path / "file.txt";
    put(file, "shared\n");
    const ObjectStore peer(dir.path / "peer");
    const auto stored = peer.store(file, 1024).value();
    const ObjectStore store(dir.path / "mine", [&] { return std::vector{peer.directory()}; });

    REQUIRE(store.store(file, 1024));
    CHECK(inode_of(store.directory() / stored.image.hash) == inode_of(peer.directory() / stored.image.hash));
    // Each store's sweep removes only its own name.
    peer.sweep({});
    CHECK(store.read(stored.image.hash, 1024).value_or("") == "shared\n");
}

TEST_CASE("A cloned file is a copy that later writes to either leave alone", "[utils][clone]") {
    TempDir dir;
    put(dir.path / "source", "original");
    const auto cloned = core::utils::clone_file(dir.path / "source", dir.path / "clone");
    if (!cloned) {
        CHECK(core::utils::clone_unsupported(cloned.error()));
        CHECK_FALSE(fs::exists(dir.path / "clone"));
        SKIP("This filesystem cannot clone files.");
    }
    put(dir.path / "source", "changed");
    CHECK(core::utils::read_file(dir.path / "clone") == "original");
    // The target must not exist: a clone never overwrites.
    CHECK_FALSE(core::utils::clone_file(dir.path / "source", dir.path / "clone"));
    CHECK(core::utils::read_file(dir.path / "clone") == "original");
}
