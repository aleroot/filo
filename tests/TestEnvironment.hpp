#pragma once

#include <cstdlib>
#include <optional>
#include <string>
#include <utility>

namespace test_support {

/// Sets an environment variable for the enclosing scope, then restores its
/// previous value, or unsets it when it had none.
class ScopedEnvVar {
public:
    ScopedEnvVar(std::string name, const std::string& value) : name_(std::move(name)) {
        if (const char* existing = std::getenv(name_.c_str())) previous_ = existing;
        ::setenv(name_.c_str(), value.c_str(), 1);
    }

    ~ScopedEnvVar() {
        if (previous_) {
            ::setenv(name_.c_str(), previous_->c_str(), 1);
        } else {
            ::unsetenv(name_.c_str());
        }
    }

    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

private:
    std::string name_;
    std::optional<std::string> previous_;
};

} // namespace test_support
