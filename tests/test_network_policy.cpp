#include <catch2/catch_test_macros.hpp>

#include "core/net/EgressScope.hpp"
#include "core/policy/ApprovalChannel.hpp"
#include "core/policy/PolicyRules.hpp"
#include "core/net/NetworkAddress.hpp"
#include "core/policy/PolicySnapshot.hpp"
#include "core/policy/PolicySource.hpp"

#include <arpa/inet.h>
#include <cpr/cpr.h>
#include <chrono>
#include <netinet/in.h>
#include <stdexcept>

using namespace core::policy;

namespace {

Subject web_tool_subject(std::string id = "fetch_url") {
    return Subject{.kind = PrincipalKind::web_tool, .id = std::move(id)};
}

Action read_action() {
    return Action{.operation = OperationKind::read, .flow = DataFlow::inbound};
}

Resource resource(std::string host, std::uint16_t port) {
    return Resource{.host = std::move(host), .port = port};
}

// A channel that records what it was asked and answers with a fixed outcome.
class ScriptedChannel final : public IApprovalChannel {
public:
    explicit ScriptedChannel(ApprovalOutcome outcome) : outcome_(outcome) {}
    ApprovalOutcome request(const ApprovalRequest& request) override {
        ++requests;
        last = request;
        return outcome_;
    }
    int requests = 0;
    ApprovalRequest last{};

private:
    ApprovalOutcome outcome_;
};

} // namespace

TEST_CASE("host patterns match exactly, by suffix and by address range",
          "[policy][network]") {
    const auto exact = HostPattern::parse("Example.COM.");
    REQUIRE(exact);
    CHECK(exact->matches("example.com"));
    CHECK(exact->matches("EXAMPLE.com"));
    CHECK_FALSE(exact->matches("www.example.com"));
    CHECK_FALSE(exact->matches(""));

    const auto suffix = HostPattern::parse("*.example.com");
    REQUIRE(suffix);
    CHECK(suffix->matches("example.com"));
    CHECK(suffix->matches("api.example.com"));
    CHECK(suffix->matches("a.b.example.com"));
    CHECK_FALSE(suffix->matches("notexample.com"));
    CHECK_FALSE(suffix->matches("example.org"));

    const auto any = HostPattern::parse("*");
    REQUIRE(any);
    CHECK(any->matches("anything.example"));

    const auto v4 = HostPattern::parse("10.0.0.0/8");
    REQUIRE(v4);
    CHECK(v4->matches("10.1.2.3"));
    CHECK_FALSE(v4->matches("11.1.2.3"));
    CHECK_FALSE(v4->matches("example.com"));

    const auto v6 = HostPattern::parse("2001:db8::/32");
    REQUIRE(v6);
    CHECK(v6->matches("2001:db8::1"));
    CHECK_FALSE(v6->matches("2001:db9::1"));
    // An IPv4-mapped address never belongs to an IPv6 range.
    CHECK_FALSE(v6->matches("::ffff:10.0.0.1"));
}

TEST_CASE("host patterns reject malformed input", "[policy][network]") {
    CHECK_FALSE(HostPattern::parse(""));
    CHECK_FALSE(HostPattern::parse("https://example.com"));
    CHECK_FALSE(HostPattern::parse("10.0.0.0/33"));
    CHECK_FALSE(HostPattern::parse("10.0.0.0/x"));
    CHECK_FALSE(HostPattern::parse("*. "));
    CHECK_FALSE(HostPattern::parse(".."));
}

TEST_CASE("port ranges cover single ports, ranges and every port", "[policy][network]") {
    const auto exact = PortRange::parse("443");
    REQUIRE(exact);
    CHECK(exact->contains(443));
    CHECK_FALSE(exact->contains(80));

    const auto range = PortRange::parse("8000-9000");
    REQUIRE(range);
    CHECK(range->contains(8000));
    CHECK(range->contains(9000));
    CHECK_FALSE(range->contains(9001));

    const auto any = PortRange::parse("*");
    REQUIRE(any);
    CHECK(any->contains(1));
    CHECK(any->contains(65535));

    CHECK_FALSE(PortRange::parse("0"));
    CHECK_FALSE(PortRange::parse("70000"));
    CHECK_FALSE(PortRange::parse("9000-8000"));
    CHECK_FALSE(PortRange::parse("http"));
}

TEST_CASE("the narrowest matching rule decides and ties resolve conservatively",
          "[policy][network]") {
    PolicyRule broad{.id = "broad", .effect = Effect::deny};
    PolicyRule narrow{.id = "narrow",
                       .effect = Effect::allow,
                       .host = HostPattern::parse("*.example.com").value()};
    const PolicySet policy({broad, narrow});
    REQUIRE(policy.valid());
    CHECK(policy.evaluate(web_tool_subject(), read_action(), resource("api.example.com", 443))
              .effect == Effect::allow);
    CHECK(policy.evaluate(web_tool_subject(), read_action(), resource("other.test", 443))
              .effect == Effect::deny);

    // Equal specificity: deny beats ask beats allow.
    PolicyRule allow{.id = "allow", .effect = Effect::allow,
                     .host = HostPattern::parse("example.com").value()};
    PolicyRule ask{.id = "ask", .effect = Effect::ask,
                   .host = HostPattern::parse("example.com").value()};
    PolicyRule deny{.id = "deny", .effect = Effect::deny,
                    .host = HostPattern::parse("example.com").value()};
    CHECK(PolicySet({allow, ask}).evaluate(web_tool_subject(), read_action(),
                                            resource("example.com", 443)).effect == Effect::ask);
    CHECK(PolicySet({allow, ask, deny}).evaluate(web_tool_subject(), read_action(),
                                                  resource("example.com", 443)).effect == Effect::deny);
    CHECK(PolicySet({deny, ask, allow}).evaluate(web_tool_subject(), read_action(),
                                                  resource("example.com", 443)).effect == Effect::deny);
}

TEST_CASE("a rule narrows by principal, operation and flow", "[policy][network]") {
    PolicyRule upload{.id = "no-upload",
                      .effect = Effect::deny,
                      .operation = OperationKind::write,
                      .flow = DataFlow::outbound};
    const PolicySet policy({upload}, Effect::allow);
    const auto read = policy.evaluate(web_tool_subject(), read_action(),
                                      resource("example.com", 443));
    CHECK(read.effect == Effect::allow);
    const auto write = policy.evaluate(
        web_tool_subject(),
        Action{.operation = OperationKind::write, .flow = DataFlow::outbound},
        resource("example.com", 443));
    CHECK(write.effect == Effect::deny);
    CHECK(write.rule_id == "no-upload");
}

TEST_CASE("an unmatched request falls back to the policy default", "[policy][network]") {
    const PolicySet policy({}, Effect::ask);
    const auto decision = policy.evaluate(web_tool_subject(), read_action(),
                                          resource("example.com", 443));
    CHECK(decision.effect == Effect::ask);
    CHECK(decision.rule_id.empty());
}

TEST_CASE("a policy with duplicate rule identifiers fails closed", "[policy][network]") {
    PolicyRule first{.id = "same", .effect = Effect::allow};
    PolicyRule second{.id = "same", .effect = Effect::deny};
    const PolicySet policy({first, second});
    CHECK_FALSE(policy.valid());
    CHECK(policy.evaluate(web_tool_subject(), read_action(), resource("example.com", 443))
              .effect == Effect::deny);
}

TEST_CASE("grants authorize exactly one destination and expire", "[policy][network]") {
    GrantStore grants;
    const auto subject = web_tool_subject();
    const auto action = read_action();
    const auto target = resource("example.com", 443);
    CHECK_FALSE(grants.consume(subject, action, target, "chat-a"));

    grants.issue(PolicyGrant{.session_id = "chat-a",
                              .subject = subject,
                              .action = action,
                              .resource = target,
                              .expires_at = std::chrono::steady_clock::now() + std::chrono::minutes(5),
                              .uses = 1});
    CHECK(grants.consume(subject, action, target, "chat-a"));
    // A single-use grant is consumed by that use.
    CHECK_FALSE(grants.consume(subject, action, target, "chat-a"));
    // A grant never covers another destination or another principal.
    grants.issue(PolicyGrant{.session_id = "chat-a",
                              .subject = subject,
                              .action = action,
                              .resource = target,
                              .expires_at = std::chrono::steady_clock::now() + std::chrono::minutes(5),
                              .uses = 0});
    CHECK(grants.consume(subject, action, resource("other.example", 443), "chat-a") == false);
    CHECK(grants.consume(web_tool_subject("other_tool"), action, target, "chat-a") == false);
    CHECK(grants.consume(subject, action, target, "chat-a"));

    grants.revoke_all();
    CHECK(grants.size() == 0);

    // An expired grant authorizes nothing.
    GrantStore expiring;
    expiring.issue(PolicyGrant{.session_id = "chat-a",
                              .subject = subject,
                                .action = action,
                                .resource = target,
                                .expires_at = std::chrono::steady_clock::now()
                                             - std::chrono::seconds(1),
                                .uses = 0});
    CHECK_FALSE(expiring.consume(subject, action, target, "chat-a"));
}

TEST_CASE("a destination the user configured replaces the default and nothing else",
          "[policy][network]") {
    const auto intent = NetworkIntent{.subject = web_tool_subject(),
                                      .action = read_action(),
                                      .resource = resource("example.com", 443)};
    const ConfiguredDestination configured{
        .kind = PrincipalKind::web_tool,
        .id = "fetch_url",
        .operation = OperationKind::read,
        .flow = DataFlow::inbound,
        .resource = intent.resource};

    // Default ask, and nobody to ask: the configured destination is still
    // allowed, and the channel is not called.
    auto channel = std::make_shared<ScriptedChannel>(ApprovalOutcome::denied);
    PolicySnapshot asking(PolicySet({}, Effect::ask), std::make_shared<GrantStore>(), channel);
    const auto allowed = asking.authorize(intent, &configured);
    CHECK(allowed.effect == Effect::allow);
    CHECK(channel->requests == 0);

    // Default deny also blocks a configured destination.
    PolicySnapshot denying(PolicySet({}, Effect::deny));
    CHECK(denying.authorize(intent, &configured).effect == Effect::deny);

    PolicyRule allow;
    allow.id = "allow-example";
    allow.effect = Effect::allow;
    allow.host = HostPattern::exact("example.com");
    PolicySnapshot explicit_allow(PolicySet({allow}, Effect::deny));
    CHECK(explicit_allow.authorize(intent, &configured).effect == Effect::allow);

    // An explicit deny still wins, and so does an explicit ask.
    PolicyRule deny;
    deny.id = "deny-example";
    deny.effect = Effect::deny;
    deny.host = HostPattern::exact("example.com");
    PolicySnapshot explicit_deny(PolicySet({deny}, Effect::ask), {}, channel);
    CHECK(explicit_deny.authorize(intent, &configured).effect == Effect::deny);
    CHECK(channel->requests == 0);

    PolicyRule ask;
    ask.id = "ask-example";
    ask.effect = Effect::ask;
    ask.host = HostPattern::exact("example.com");
    PolicySnapshot explicit_ask(PolicySet({ask}, Effect::deny), {}, channel);
    CHECK(explicit_ask.authorize(intent, &configured).effect == Effect::deny);
    CHECK(channel->requests == 1);

    // An unnamed deny is still a rule. An empty label must not look like the
    // policy default and let the configured destination through.
    PolicyRule unnamed;
    unnamed.effect = Effect::deny;
    unnamed.host = HostPattern::exact("example.com");
    PolicySnapshot unnamed_deny(PolicySet({unnamed}, Effect::allow), {}, channel);
    CHECK(unnamed_deny.authorize(intent, &configured).effect == Effect::deny);
    CHECK(channel->requests == 1);

    // A different host, principal or flow is not the destination that was
    // configured, so the default stands.
    const auto other = NetworkIntent{.subject = web_tool_subject(),
                                     .action = read_action(),
                                     .resource = resource("other.example", 443)};
    CHECK(asking.authorize(other, &configured).effect == Effect::deny);
    CHECK(channel->requests == 2);
    const auto other_tool = NetworkIntent{.subject = web_tool_subject("web_search"),
                                          .action = read_action(),
                                          .resource = intent.resource};
    CHECK(denying.authorize(other_tool, &configured).effect == Effect::deny);
}

TEST_CASE("an asked request is answered by the approval channel", "[policy][network]") {
    const auto intent = NetworkIntent{.subject = web_tool_subject(),
                                      .action = read_action(),
                                      .resource = resource("example.com", 443)};

    auto approve = std::make_shared<ScriptedChannel>(ApprovalOutcome::approved_once);
    PolicySnapshot snapshot(PolicySet({}, Effect::ask), std::make_shared<GrantStore>(), approve);
    const auto decision = snapshot.authorize(intent);
    CHECK(decision.effect == Effect::allow);
    CHECK(approve->requests == 1);
    CHECK(approve->last.resource.host == "example.com");
    CHECK(approve->last.description.find("example.com") != std::string::npos);

    auto deny = std::make_shared<ScriptedChannel>(ApprovalOutcome::denied);
    PolicySnapshot denied(PolicySet({}, Effect::ask), std::make_shared<GrantStore>(), deny);
    CHECK(denied.authorize(intent).effect == Effect::deny);
    CHECK(deny->requests == 1);

    // A channel that cannot ask anyone denies: an ask is not an allow.
    PolicySnapshot headless(PolicySet({}, Effect::ask));
    CHECK(headless.authorize(intent).effect == Effect::deny);

    // A channel that throws cannot authorize and must not leak its text.
    class ThrowingChannel final : public IApprovalChannel {
    public:
        ApprovalOutcome request(const ApprovalRequest&) override {
            throw std::runtime_error("secret token");
        }
    };
    PolicySnapshot failing(PolicySet({}, Effect::ask), std::make_shared<GrantStore>(),
                           std::make_shared<ThrowingChannel>());
    const auto failed = failing.authorize(intent);
    CHECK(failed.effect == Effect::deny);
    CHECK(failed.reason.find("secret") == std::string::npos);
}

TEST_CASE("standing grants are scoped to a live chat", "[policy][network]") {
    const auto intent = NetworkIntent{.subject = web_tool_subject(),
                                      .action = read_action(),
                                      .resource = resource("example.com", 443)};
    auto approve = std::make_shared<ScriptedChannel>(ApprovalOutcome::approved_for_session);
    auto grants = std::make_shared<GrantStore>();
    PolicySnapshot snapshot(PolicySet({}, Effect::ask), grants, approve);
    {
        ApprovalOriginScope chat("chat-a", {});
        CHECK(snapshot.authorize(intent).effect == Effect::allow);
        CHECK(snapshot.authorize(intent).effect == Effect::allow);
        CHECK(approve->requests == 1);
    }
    {
        ApprovalOriginScope chat("chat-b", {});
        CHECK(snapshot.authorize(intent).effect == Effect::allow);
        CHECK(approve->requests == 2);
    }
    grants->revoke_session("chat-a");
    {
        ApprovalOriginScope chat("chat-a", {});
        CHECK(snapshot.authorize(intent).effect == Effect::allow);
        CHECK(approve->requests == 3);
    }
    {
        ApprovalOriginScope chat("chat-b", {});
        CHECK(snapshot.authorize(intent).effect == Effect::allow);
        CHECK(approve->requests == 3);
    }
    // No chat may reuse a standing grant or create one.
    CHECK(snapshot.authorize(intent).effect == Effect::allow);
    CHECK(snapshot.authorize(intent).effect == Effect::allow);
    CHECK(approve->requests == 5);
}

TEST_CASE("a policy document is parsed strictly", "[policy][network]") {
    const auto policy = parse_policy_document(R"({
        "version": 1,
        "default": "deny",
        "rules": [
            { "id": "read-github",
              "effect": "allow",
              "principal": "web_tool:fetch_url",
              "operation": "read",
              "flow": "inbound",
              "host": "*.github.com",
              "port": 443 },
            { "effect": "deny", "principal": "mcp_server", "host": "*", "port": "*" }
        ]
    })");
    REQUIRE(policy);
    CHECK(policy->size() == 2);
    CHECK(policy->default_effect() == Effect::deny);
    CHECK(policy->evaluate(web_tool_subject(), read_action(), resource("api.github.com", 443))
              .effect == Effect::allow);
    CHECK(policy->evaluate(Subject{.kind = PrincipalKind::mcp_server, .id = "server"},
                           Action{.operation = OperationKind::invoke,
                                  .flow = DataFlow::bidirectional},
                           resource("mcp.example", 443)).effect == Effect::deny);

    // Unknown fields, unknown values and malformed ports are errors: a policy
    // that cannot be understood must never be treated as an absent policy.
    CHECK_FALSE(parse_policy_document(R"({"rules": [], "nonsense": 1})"));
    CHECK_FALSE(parse_policy_document(R"({"rules": [{"effect": "maybe"}]})"));
    CHECK_FALSE(parse_policy_document(R"({"rules": [{"effect": "allow", "port": 0}]})"));
    CHECK_FALSE(parse_policy_document(R"({"rules": [{"effect": "allow", "extra": 1}]})"));
    CHECK_FALSE(parse_policy_document(R"({"rules": [{"effect": "allow", "principal": "nope"}]})"));
    CHECK_FALSE(parse_policy_document("not json"));
    CHECK_FALSE(parse_policy_document(R"({"version": 2})"));
    // A duplicate rule identifier is a configuration error, not a policy.
    CHECK_FALSE(parse_policy_document(
        R"({"rules": [{"id": "a", "effect": "allow"}, {"id": "a", "effect": "deny"}]})"));
}

TEST_CASE("a policy field of the wrong type is an error, never a wildcard", "[policy][network]") {
    // An omitted field matches everything, so a field that is present but
    // unreadable must not be mistaken for an omitted one: `"host": 123`
    // would otherwise allow every host.
    for (const auto* document : {
             R"({"rules": [{"effect": "allow", "host": 123}]})",
             R"({"rules": [{"effect": "allow", "principal": 7}]})",
             R"({"rules": [{"effect": "allow", "operation": true}]})",
             R"({"rules": [{"effect": "allow", "flow": []}]})",
             R"({"rules": [{"effect": "allow", "host": null}]})",
             R"({"rules": [{"effect": "allow", "id": 5}]})",
             R"({"rules": [{"effect": "allow", "port": 443.5}]})",
             R"({"rules": [{"effect": "allow", "port": true}]})",
             R"({"rules": [{"effect": 1}]})",
             R"({"version": "1", "rules": []})",
             R"({"default": 0, "rules": []})",
         }) {
        INFO(document);
        CHECK_FALSE(parse_policy_document(document));
    }
    const auto host = parse_policy_document(R"({"rules": [{"effect": "allow", "host": 123}]})");
    REQUIRE_FALSE(host);
    CHECK(host.error().find("\"host\"") != std::string::npos);
}

TEST_CASE("a question names the chat that asked, and only that chat", "[policy][network]") {
    auto channel = std::make_shared<ScriptedChannel>(ApprovalOutcome::denied);
    const PolicySnapshot snapshot(PolicySet({}, Effect::ask), std::make_shared<GrantStore>(), channel);
    const NetworkIntent intent{.subject = web_tool_subject(),
                               .action = read_action(),
                               .resource = resource("example.com", 443)};

    // Outside any scope a question names no chat, so no interface shows it.
    (void)snapshot.authorize(intent);
    REQUIRE(channel->requests == 1);
    CHECK(channel->last.origin.session_id.empty());
    CHECK_FALSE(channel->last.origin.stopped);

    bool chat_a_stopped = false;
    {
        const ApprovalOriginScope chat_a("chat-a", [&] { return chat_a_stopped; });
        {
            const ApprovalOriginScope chat_b("chat-b", {});
            (void)snapshot.authorize(intent);
            CHECK(channel->last.origin.session_id == "chat-b");
        }
        // The inner work ended, so the outer chat is named again.
        (void)snapshot.authorize(intent);
        CHECK(channel->last.origin.session_id == "chat-a");
        // The question follows that chat's own stop state.
        REQUIRE(channel->last.origin.stopped);
        CHECK_FALSE(channel->last.origin.stopped());
        chat_a_stopped = true;
        CHECK(channel->last.origin.stopped());
    }
    CHECK(ApprovalOriginScope::current() == nullptr);
}

TEST_CASE("a scope decides a destination once and binds the request to it",
          "[policy][network]") {
    // Without a policy the scope decides nothing and leaves the request alone.
    {
        core::net::EgressScope scope({}, web_tool_subject(), read_action());
        CHECK_FALSE(scope.enforcing());
        cpr::Session unscoped;
        unscoped.SetUrl(cpr::Url{"https://api.example.com/v1/files"});
        CHECK_FALSE(scope.admit(unscoped));
        CHECK_FALSE(core::net::admit_in_scope(unscoped));
        CHECK_FALSE(core::net::binding_required());
    }

    auto policy = std::make_shared<PolicySnapshot>(PolicySet({}, Effect::allow),
                                                   std::make_shared<GrantStore>());
    core::net::EgressScope scope(policy, web_tool_subject(), read_action(),
                                 "https://api.example.com/v1");
    REQUIRE(scope.enforcing());
    CHECK(core::net::binding_required());

    // The configured endpoint is allowed, and the decision is taken once: a
    // second request to it is not asked about again.
    CHECK_FALSE(scope.authorize("https://api.example.com/v1/files"));
    cpr::Session same;
    same.SetUrl(cpr::Url{"https://api.example.com/v1/files"});
    CHECK_FALSE(scope.admit(same));

    // Another host on the same port is a different destination.
    auto other_host = std::make_shared<PolicySnapshot>(
        PolicySet({}, Effect::deny), std::make_shared<GrantStore>());
    core::net::EgressScope refusing(other_host, web_tool_subject(), read_action(),
                                    "https://api.example.com/v1");
    CHECK(refusing.authorize("https://upload.example.net/v1/files").has_value());
    cpr::Session other;
    other.SetUrl(cpr::Url{"https://upload.example.net/v1/files"});
    CHECK(refusing.admit(other).has_value());
}

TEST_CASE("a scope in effect on the thread admits the requests made inside it",
          "[policy][network]") {
    auto policy = std::make_shared<PolicySnapshot>(PolicySet({}, Effect::allow),
                                                   std::make_shared<GrantStore>());
    cpr::Session session;
    session.SetUrl(cpr::Url{"https://api.example.com/v1/files"});
    {
        core::net::EgressScope scope(policy, web_tool_subject(), read_action());
        CHECK_FALSE(core::net::admit_in_scope(session));
    }
    // Outside the scope the request is left as it was built.
    cpr::Session outside;
    outside.SetUrl(cpr::Url{"https://api.example.com/v1/files"});
    CHECK_FALSE(core::net::admit_in_scope(outside));
    CHECK_FALSE(core::net::binding_required());
}

TEST_CASE("network address checks bind resolved IPs to safe or explicit targets",
          "[policy][network]") {
    const auto v4 = [](const char* text, std::string_view host) {
        sockaddr_in address{};
        address.sin_family = AF_INET;
        REQUIRE(::inet_pton(AF_INET, text, &address.sin_addr) == 1);
        return core::net::permits_resolved_address(reinterpret_cast<const sockaddr*>(&address),
                                                    sizeof(address), host);
    };
    const auto v6 = [](const char* text, std::string_view host) {
        sockaddr_in6 address{};
        address.sin6_family = AF_INET6;
        REQUIRE(::inet_pton(AF_INET6, text, &address.sin6_addr) == 1);
        return core::net::permits_resolved_address(reinterpret_cast<const sockaddr*>(&address),
                                                    sizeof(address), host);
    };
    CHECK(v4("8.8.8.8", "example.com"));
    CHECK_FALSE(v4("127.0.0.1", "example.com"));
    CHECK(v4("127.0.0.1", "127.0.0.1"));
    CHECK_FALSE(v4("10.1.2.3", "example.com"));
    CHECK_FALSE(v4("169.254.169.254", "169.254.169.254"));
    CHECK_FALSE(v4("100.100.100.200", "100.100.100.200"));
    CHECK(v6("2606:4700:4700::1111", "example.com"));
    CHECK_FALSE(v6("::1", "example.com"));
    CHECK(v6("::1", "::1"));
    CHECK_FALSE(v6("fd00:ec2::254", "fd00:ec2::254"));
    CHECK_FALSE(v6("::ffff:169.254.169.254", "::ffff:169.254.169.254"));
    CHECK_FALSE(core::net::permits_resolved_address(nullptr, 0, "example.com"));
    // `localhost` names (RFC 6761) bind to loopback, and only to loopback.
    CHECK(v4("127.0.0.1", "localhost"));
    CHECK(v6("::1", "localhost"));
    CHECK(v4("127.0.0.1", "api.localhost"));
    CHECK_FALSE(v4("10.1.2.3", "localhost"));
    CHECK_FALSE(v4("8.8.8.8", "localhost"));
    CHECK_FALSE(v6("2606:4700:4700::1111", "localhost"));
    CHECK_FALSE(v4("169.254.169.254", "localhost"));
    CHECK_FALSE(v4("127.0.0.1", "notlocalhost"));
    CHECK_FALSE(v4("127.0.0.1", "localhost.example.com"));
}
