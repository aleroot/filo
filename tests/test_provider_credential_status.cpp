#include <catch2/catch_test_macros.hpp>

#include "core/auth/ApiKeyCredentialSource.hpp"
#include "core/auth/ICredentialSource.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/llm/ProviderCredentialStatus.hpp"

#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

using core::auth::CredentialAvailability;
using core::llm::ProviderCredentialState;

namespace {

/** Credential source that answers the availability probe with a fixed value. */
class FixedAvailabilitySource final : public core::auth::ICredentialSource {
public:
    explicit FixedAvailabilitySource(CredentialAvailability availability)
        : availability_(availability) {}

    core::auth::AuthInfo get_auth() override { return {}; }
    CredentialAvailability availability() override { return availability_; }

private:
    CredentialAvailability availability_;
};

/** Credential source that fails the probe, standing in for a broken store. */
class ThrowingAvailabilitySource final : public core::auth::ICredentialSource {
public:
    core::auth::AuthInfo get_auth() override { return {}; }
    CredentialAvailability availability() override {
        throw std::runtime_error("token store unreadable");
    }
};

class StubProvider final : public core::llm::LLMProvider {
public:
    StubProvider() = default;

    void stream_response(
        const core::llm::ChatRequest&,
        std::function<void(const core::llm::StreamChunk&)>) override {}

    [[nodiscard]] core::llm::ProviderCapabilities capabilities() const override {
        return capabilities_;
    }

    [[nodiscard]] std::optional<core::llm::ProviderMetadata> metadata()
        const override {
        return metadata_;
    }

    void set_local(bool local) { capabilities_.is_local = local; }
    void set_metadata(std::optional<core::llm::ProviderMetadata> metadata) {
        metadata_ = std::move(metadata);
    }

private:
    core::llm::ProviderCapabilities capabilities_{};
    std::optional<core::llm::ProviderMetadata> metadata_;
};

[[nodiscard]] std::shared_ptr<StubProvider> provider_with(
    std::shared_ptr<core::auth::ICredentialSource> source) {
    auto provider = std::make_shared<StubProvider>();
    provider->set_metadata(core::llm::ProviderMetadata{
        .credential_source = std::move(source),
    });
    return provider;
}

} // namespace

TEST_CASE("An API key source reports whether it can authenticate a request",
          "[llm][credentials]") {
    // The factories drop the header entirely for an empty key, so an empty
    // AuthInfo is exactly the state that produces an unauthenticated request.
    REQUIRE(core::auth::ApiKeyCredentialSource::as_bearer("")->availability()
            == CredentialAvailability::Missing);
    REQUIRE(core::auth::ApiKeyCredentialSource::none()->availability()
            == CredentialAvailability::Missing);
    REQUIRE(core::auth::ApiKeyCredentialSource::as_bearer("sk-live")->availability()
            == CredentialAvailability::Ready);
    REQUIRE(core::auth::ApiKeyCredentialSource::as_query_param("k")->availability()
            == CredentialAvailability::Ready);
    REQUIRE(core::auth::ApiKeyCredentialSource::as_custom_header("k", "x-api-key")
                ->availability()
            == CredentialAvailability::Ready);
}

TEST_CASE("Provider credential state is a safe, non-blocking probe",
          "[llm][credentials]") {
    SECTION("a missing key is reported as missing") {
        const auto provider = provider_with(
            core::auth::ApiKeyCredentialSource::as_bearer(""));
        REQUIRE(core::llm::provider_credential_state(provider)
                == ProviderCredentialState::Missing);
        REQUIRE(core::llm::credentials_missing(
            core::llm::provider_credential_state(provider)));
    }

    SECTION("a configured key is reported as ready") {
        const auto provider = provider_with(
            core::auth::ApiKeyCredentialSource::as_bearer("sk-live"));
        REQUIRE(core::llm::provider_credential_state(provider)
                == ProviderCredentialState::Ready);
    }

    SECTION("local backends need no credential") {
        auto provider = provider_with(
            core::auth::ApiKeyCredentialSource::as_bearer(""));
        provider->set_local(true);
        REQUIRE(core::llm::provider_credential_state(provider)
                == ProviderCredentialState::Ready);
    }

    SECTION("providers without metadata or a source stay unknown") {
        auto without_metadata = std::make_shared<StubProvider>();
        REQUIRE(core::llm::provider_credential_state(without_metadata)
                == ProviderCredentialState::Unknown);

        const auto without_source = provider_with(nullptr);
        REQUIRE(core::llm::provider_credential_state(without_source)
                == ProviderCredentialState::Unknown);

        REQUIRE(core::llm::provider_credential_state(nullptr)
                == ProviderCredentialState::Unknown);
    }

    SECTION("a source that cannot answer never demotes the provider") {
        const auto unknown = provider_with(
            std::make_shared<FixedAvailabilitySource>(
                CredentialAvailability::Unknown));
        REQUIRE(core::llm::provider_credential_state(unknown)
                == ProviderCredentialState::Unknown);
        REQUIRE_FALSE(core::llm::credentials_missing(
            core::llm::provider_credential_state(unknown)));

        const auto throwing =
            provider_with(std::make_shared<ThrowingAvailabilitySource>());
        REQUIRE(core::llm::provider_credential_state(throwing)
                == ProviderCredentialState::Unknown);
    }
}

TEST_CASE("Credential ordering keeps usable services first without hiding any",
          "[llm][credentials]") {
    SECTION("missing entries sink to the end, everything else keeps its order") {
        const std::vector<ProviderCredentialState> states{
            ProviderCredentialState::Missing,   // 0: public endpoint, no key
            ProviderCredentialState::Ready,     // 1: token plan, key present
            ProviderCredentialState::Missing,   // 2: coding plan, no key
            ProviderCredentialState::Unknown,   // 3: cannot tell
            ProviderCredentialState::Ready,     // 4
        };

        const auto order = core::llm::order_by_credential_state(states);

        REQUIRE(order == std::vector<std::size_t>{1, 3, 4, 0, 2});
    }

    SECTION("a fully unauthenticated catalogue is left untouched") {
        const std::vector<ProviderCredentialState> states(
            3, ProviderCredentialState::Missing);

        const auto order = core::llm::order_by_credential_state(states);

        REQUIRE(order == std::vector<std::size_t>{0, 1, 2});
    }

    SECTION("an empty catalogue produces an empty ordering") {
        REQUIRE(core::llm::order_by_credential_state({}).empty());
    }
}

TEST_CASE("Only a missing credential is annotated in the picker",
          "[llm][credentials]") {
    REQUIRE(core::llm::credential_state_note(ProviderCredentialState::Ready)
            .empty());
    REQUIRE(core::llm::credential_state_note(ProviderCredentialState::Unknown)
            .empty());
    REQUIRE_FALSE(
        core::llm::credential_state_note(ProviderCredentialState::Missing)
            .empty());
}
