# MiniMax M Plan support in Filo

Research baseline: 21 October 2026. Filo revision: `33c1cc6`. MiniMax Code reference revision: `da22f11c55d4748155d7162c1030414228a00565`. The implementation now extracts `ClaudeProtocol` and `ClaudeModelTraits` from the neutral Messages layer and adds MiniMax M Plan/API presets, login flows, model discovery, reasoning/tool continuation, scoped session state, image input, context admission/native counting, quota reporting, and error handling. A small `MiniMaxProvider` supplies compaction estimates from the same budget policy used by `MiniMaxProtocol`; transport remains shared. `context_window: 1000000` explicitly opts into the larger working context. The design and staged acceptance plan below record the research decisions. Live account entitlement, billing, and cache-hit behavior still require a Subscription Key.

Offline verification after regression review: all 3,343 unit cases (114,089 assertions) and 60 provider HTTP/web/WebSocket integration cases passed, including per-account quota concurrency and bounded token-count fallback coverage. Filo and both test binaries build successfully. Socket and subprocess tests were run with local test permissions after the sandbox blocked them. No live MiniMax account/billing verification was performed.

MiniMax M Plan can be used from third-party coding clients through a Subscription Key. Filo now supplies native MiniMax registration, authentication, model cards, discovery, request policy, and subscription accounting. Native Messages support separates the shared wire format from each vendor's product behavior. [Subscription setup](https://platform.minimax.io/docs/m-plan/quickstart), [third-party tool configuration](https://platform.minimax.io/docs/m-plan/other-tools).

**1. Correct the existing protocol boundary before adding MiniMax.**

The user's architectural concern is correct: `AnthropicProtocol` currently combines Messages serialization with Claude product behavior. This is an existing separation-of-responsibilities problem, not a MiniMax-specific compatibility requirement.

Concrete evidence in `src/core/llm/protocols/AnthropicProtocol.cpp`:

- `compose_anthropic_system_prompt()` and `append_anthropic_system_field()` insert a Claude billing marker into the system prompt, including when the caller supplied no system prompt.
- `build_headers()` adds a Claude billing header and Claude Code beta features by default.
- `prepare_request()` normalizes Claude aliases and the `[1m]` model decoration.
- The default reasoning emitter resolves Claude generation policies; response enrichment can query Claude account usage.
- `AnthropicWebSearchBackend.cpp` calls the serializer directly, so moving behavior requires updating this caller too.

The intended dependency structure is:

```text
HttpLLMProvider                 HTTP, cancellation, bounded retries
  AnthropicProtocol            Messages wire encoding, SSE, tools, usage
    ClaudeProtocol             Claude models, account behavior, extensions
    MiniMaxProtocol            MiniMax models, request policy, errors, quota
    ZaiCodingProtocol          Existing GLM Coding Plan adapter
```

Keep `ApiType::Anthropic` as the wire-format enum. Do not add `ApiType::MiniMax` or infer the vendor from the wire type. Introduce `ClaudeProtocol.hpp/.cpp`, move Claude policy there, and make the shared serializer vendor-neutral by default. Give it explicit serialization policy/options where needed; do not make an empty extension callback mean “apply Claude behavior.”

The shared layer owns system-message representation, tool/result encoding, content-block order, fragmented SSE parsing, usage fields, terminal events, and lossless continuation storage. It can represent optional protocol fields without deciding which models accept them. Claude owns alias resolution, its reasoning and sampling rules, billing attribution, OAuth/beta headers, signed-prefix extensions, hosted tools, and account usage requests. MiniMax and Z.ai supply their own policies.

Route built-in Claude and Claude OAuth configurations to `ClaudeProtocol`; custom `api_type: anthropic` configurations receive the clean shared protocol unless they explicitly select a vendor adapter. Preserve explicit Claude compatibility for existing custom configurations that intentionally target native Anthropic, using a documented migration/selection rule rather than guessing from every model name. Move Claude policy helpers out of the shared module; `AnthropicCompatibility.hpp` is currently largely Claude model policy and should become clearly Claude-owned.

The `ContinuationItem.provider = "anthropic"` label is documented as a wire-protocol owner. The label alone is not proof of a defect. Audit the full replay path for originating service, model, and endpoint scope. Add explicit provenance where absent, while retaining the wire-format label if useful. Legacy sessions must continue to load; ambiguous legacy continuation must not be forwarded automatically to another vendor. Preserve ordinary conversation text and valid tool history when opaque continuation is ineligible for replay.

Acceptance: a plain Messages request contains no injected Claude text, aliases, billing headers, or Claude account traffic. Existing Claude behavior remains covered by Claude-specific tests, Z.ai remains independent, and adapter clones retain their concrete type and configuration.

**2. Reuse the right patterns from Xiaomi and Qwen.**

| Existing Filo implementation | What to reuse for MiniMax |
| --- | --- |
| `ProviderDefinition.hpp` | Declarative endpoints, ordered credential lookup, subscription billing, dashboard, provider-name boundaries. |
| `MimoProtocol` and `MimoModelTraits` | Small vendor adapter and centralized vendor facts, while sharing transport. |
| `MimoAuthenticationStrategy` | Distinct plan/API choices and regional awareness; MiniMax does not need MiMo's OAuth implementation. |
| `QwenProvider`, `QwenRequestBudget`, `DashScopeProtocol` | Separate request policy, safe budgeting, cancellation, isolated parallel state, and bounded repair when evidence warrants it. |
| `ProviderCatalogGrouping.cpp` | Group a vendor visually while preserving each credential/billing source as a routing identity. |
| `ZaiCodingProtocol` | Existing example of a vendor using Messages with its own reasoning contract; migrate it onto the cleaned shared layer. |

Do not copy Qwen's entire specialized provider automatically. Start MiniMax with `HttpLLMProvider` and `MiniMaxProtocol`; add a small `MiniMaxProvider` only for request-wide budgeting/token estimation that the protocol interface cannot express.

**3. Register subscription and metered access explicitly.**

Proposed names and defaults:

| Configuration | M Plan | Pay-as-you-go |
| --- | --- | --- |
| Provider key | `minimax-m-plan` | `minimax` |
| Wire type | `anthropic` | `anthropic` |
| Global base URL | `https://api.minimax.io/anthropic` | Same |
| Credential | M Plan Subscription Key | Regular API Key |
| Canonical Filo environment variable | `MINIMAX_M_PLAN_API_KEY` | `MINIMAX_API_KEY` |
| Optional plan-key alias | `MINIMAX_SUBSCRIPTION_KEY` | — |
| Billing kind | `Subscription` | `Metered` |
| Initial model | `MiniMax-M3.1-Flash` | Confirm from current catalog/account access |

Register the more specific prefix before `minimax`. Both use the same global host, so host matching cannot distinguish their billing relationship. Do not silently fall back from the plan variable to the metered variable. A Subscription Key consumes plan usage and available credit packs, whereas a regular key uses account balance; they are not interchangeable. [M Plan setup](https://platform.minimax.io/docs/m-plan/quickstart), [plan billing behavior](https://www.minimax.io/m-plan).

Use `ApiKeyPromptStrategy`: `filo --auth minimax` opens the M Plan Subscription Key prompt directly; `filo --auth minimax-m-plan` is an alias, and `filo --auth minimax-api` explicitly selects pay-as-you-go access. Persist through existing auth profile/config overlays. Explain where to obtain the Subscription Key. Detect clearly mismatched known key types without rejecting future unknown formats solely by prefix. Official tooling currently illustrates subscription keys with `sk-cp-`. [Official Pi setup](https://platform.minimax.io/docs/m-plan/pi).

Implement global access first for this subscription. The reference client also selects `api.minimaxi.com/anthropic` for mainland China; add an explicit regional option only with matching account/key validation. Never automatically send a global key to another regional host. No MiniMax Code browser-login or private managed gateway dependency is needed for Subscription Key access.

Touch `ProviderDefinition.hpp`, `ProviderFactory.cpp`, `ConfigManager.cpp`, `AuthenticationManager.cpp`, login/help UI, and credential-status tests. Maintain the current rule that subscription usage does not produce estimated pay-as-you-go USD totals.

**4. Implement the current MiniMax model contract.**

The current target is `MiniMax-M3.1-Flash`, released on 19 October 2026. M Plan lists it across Go, Explore, and Build. The preview name is retired and served by the stable model. Do not seed this new integration from older M2.7/highspeed defaults merely because reference fallback data still contains them. [Model releases](https://platform.minimax.io/docs/release-notes/models), [current M Plan models](https://platform.minimax.io/docs/m-plan/intro).

Add `MiniMaxModelTraits.hpp/.cpp` and a `MiniMaxProtocol` that:

- Sends `POST /anthropic/v1/messages`, with a normalized base URL that cannot duplicate `/v1` or `/messages`.
- Uses documented Messages authentication. Both Bearer and `x-api-key` appear in official examples; prefer one deterministic scheme per request, and verify catalog authentication separately.
- Maps `low`, `medium`, `high`, `xhigh`, and `max` to `output_config.effort`. Choose `max` initially to match the service default, with normal Filo effort controls available.
- Keeps M3.1 Flash thinking enabled. An explicit off request must resolve visibly to the lowest supported effort or produce a clear validation result; never send `thinking.type: disabled`. Use `adaptive` when emitting thinking configuration. Manual Claude thinking budgets are not a substitute.
- Omits Claude-only reasoning extensions and unsupported sampling/admission settings. M3.1 Flash does not support `service_tier`.
- Preserves the complete ordered assistant content across tool turns, including thinking and signatures. Display reasoning separately from answer text; do not reconstruct replay state from displayed text.

These are MiniMax adapter/model policies, not branches to add to the common Messages implementation. [Anthropic SDK compatibility](https://platform.minimax.io/docs/api-reference/text-anthropic-api).

The detailed Create Message schema currently says `tool_choice` supports only `auto`, despite the higher-level compatibility page describing tool choice more broadly. Use the stricter endpoint contract initially. Do not advertise forced tool selection or strict structured outputs without verification. Native Filo tool arguments still undergo normal validation before execution. [Create Message contract](https://platform.minimax.io/docs/api-reference/text-chat-anthropic).

If M3 is offered, give it separate traits: thinking is off when omitted and can be enabled adaptively. Do not apply M3.1's required-thinking policy to the whole MiniMax family. Do not assume API model availability implies M Plan entitlement.

**5. Add model discovery, limits, and correct source routing.**

Discover through `/anthropic/v1/models` using the selected provider's credentials. Reuse the existing parser where sufficient and enrich sparse results with MiniMax-owned metadata. The documented list response contains IDs and names without a complete capability/limit card; missing metadata must not erase curated traits. Retired entries in documentation examples must not become the recommended fallback. [Models endpoint](https://platform.minimax.io/docs/api-reference/models/anthropic/list-models).

Seed M3.1 Flash with tools, streaming, reasoning, image input, and prompt-cache support. Keep hosted search, PDF input, native code execution, and structured-output guarantees off unless separately implemented and verified.

Distinguish the model ceiling from Filo's effective working limits. MiniMax Code's checked-in catalog uses 512,000 context / 128,000 output by default, with a 1,000,000 context option. Other documentation examples use binary 512K values. Start with the reference client's conservative decimal defaults; do not silently expand to 1M when a sparse catalog refresh arrives. Document the explicit larger-context option and validate it against the account. [Pinned reference catalog](https://github.com/MiniMax-AI/minimax-code/blob/da22f11c55d4748155d7162c1030414228a00565/packages/config/src/minimax-model-catalog.ts).

Budget the complete serialized request, including system layers, tools, images, and replayed reasoning, and reserve output plus estimation margin. Count reasoning inside the generation budget. Integrate the documented `/anthropic/v1/messages/count_tokens` route where useful, with cancellation and a bounded timeout; retain a conservative local fallback. Apply admission and the output cap once, after counting, so an inflated fallback estimate cannot reject a fitting request or suppress valid output room. Use smaller live endpoint context limits for both admission and compaction. Compact before limits are exceeded. Update shared context-limit consumers such as `TokenAccounting.hpp` through model metadata rather than leaving inconsistent name-based defaults.

In `ProviderCatalogGrouping.cpp` and `SkillTurnResolver.hpp`, keep MiniMax separate from Claude even though both use `ApiType::Anthropic`. The latter currently maps that wire type directly to the Anthropic family. Resolve vendor identity from provider definitions before falling back to wire type. Preserve explicit provider choice when M Plan and metered catalogs contain the same model; no silent metered fallback on quota exhaustion.

**6. Make cache and subscription usage observable.**

MiniMax documents automatic prefix caching beginning at 512 input tokens. Keep stable tools/system prefixes stable, and parse cache-read and cache-creation usage without double-counting input. Verify two repeated-prefix requests show authoritative cache usage. Do not assume Claude-specific cache placement or retention settings are necessary for MiniMax. Its reference resolver explicitly disables long cache retention for first-party Messages routes. [Cache semantics](https://platform.minimax.io/docs/api-reference/text-prompt-caching), [reference resolver](https://github.com/MiniMax-AI/minimax-code/blob/da22f11c55d4748155d7162c1030414228a00565/packages/local-runtime-v2/src/service/model-system/resolution/local-model-resolver.ts).

Add a subscription quota client for documented `GET https://www.minimax.io/v1/token_plan/remains`; the official CLI uses the same path under its selected API base. Establish the supported global endpoint with a real Subscription Key before enabling any fallback. This is an API request, separate from the dashboard navigation URL. Scope credentials to the approved provider endpoint, bound the request, and cache/coalesce results by account/provider/region. A failed quota read must not fail an otherwise successful model response. [Official plan FAQ/API example](https://www.minimax.io/m-plan), [official CLI endpoint selection](https://github.com/MiniMax-AI/cli/blob/main/src/client/endpoints.ts).

The quota client uses per-account locks so reads for one account coalesce without blocking other accounts. Quota and token-count responses are bounded while streaming; unavailable or oversized responses remain best-effort failures.

Populate existing `RateLimitInfo`/`UsageWindow` data for five-hour and weekly usage, reset timestamps, and known exhaustion. Show unknown or stale status honestly. The two windows reset independently and are shared across MiniMax Code and other clients using the subscription. [Usage rules](https://platform.minimax.io/docs/m-plan/usage-rules).

Use explicit remaining-percent fields as authoritative when present: utilization is `(100 - remaining_percent) / 100`. Do not assume historical `*_usage_count` means consumed count. MiniMax's own CLI reconciles that ambiguous field against explicit percentages and rejects inconsistent interpretations. Add fixtures for both historical meanings, zero limits, decimal percentages, malformed responses, and timestamp units. [Official quota normalization](https://github.com/MiniMax-AI/cli/blob/main/src/utils/quota.ts).

Classify business errors as well as HTTP/SSE errors. Invalid key/model/parameters and exhausted usage need actionable terminal results; transient overload, timeout, and request-rate limits need bounded backoff. In particular, `2062` is a rate-limit condition for inference and can mean no active subscription during quota queries; do not classify every occurrence as exhausted quota. Distinguish `2056` usage exhaustion, `2057` wrong key type, `2061`/`2068` plan entitlement, `2067` exhausted usage with credit consumption off, and `2069` model retirement. Never enable credit consumption or switch keys automatically. [Error reference](https://platform.minimax.io/docs/api-reference/errorcode).

**7. Deliver in reviewable stages, with meaningful tests.**

| Stage | Deliverable | Required validation |
| --- | --- | --- |
| A | Clean Messages layer and extracted Claude adapter | Neutral request snapshots; Claude HTTP/auth/hosted-search regressions; Z.ai regressions; correct clone type; legacy session loading. |
| B | MiniMax presets, auth, traits, adapter, catalog and routing | Plan/API credential isolation; no Claude text/headers/account calls; stable model and effort mappings; sparse catalog fallback; source selection. |
| C | Budgeting, continuation and stream resilience | Thinking → tool → result → final answer round trip; persisted resume; edited/compacted history; provider/model switch isolation; fragmented SSE; malformed/incomplete tool JSON; EOF; cancellation; parallel forks. |
| D | Cache, quota, errors and image input | Usage accounting; quota normalization/TTL/account isolation; bounded retries before output only; permanent-error behavior; native image round trip. |
| E | Documentation and live acceptance | Dedicated key instructions; `/model` and CLI selection; small real coding task; dashboard reconciliation and measured cache behavior. |

Add `test_minimax_provider.cpp` and local-server `test_integration_minimax.cpp`, update `test_claude_provider.cpp`/shared protocol tests, and extend existing auth, catalog grouping, skill resolution, budget, session, reasoning, and usage-panel tests where behavior crosses boundaries. Register sources/tests in `CMakeLists.txt`. Existing Xiaomi and Qwen suites should pass without policy changes.

Do not automatically replay a request after partial output or execute unfinished tools. Preserve the existing transport's cancellation and retry guarantees. Add vendor-specific request repair only for a demonstrated recoverable rejection and bound it to one pre-output attempt; do not strip reasoning speculatively.

Native images belong in the initial complete coding integration because Filo already supports image attachments. Video and Files API uploads can follow as a separate extension: the reference has upload/body-size capabilities, but Filo needs an actual video/file path before advertising them. Official web-search MCP can be documented as optional through Filo's existing MCP system. Video generation, speech, and other M Plan products are separate from this coding-model implementation.

**8. Subscription readiness and completion criteria.**

The integration is ready for use when a real global M Plan Subscription Key completes a streaming coding turn, executes validated tools, continues with preserved thinking, resumes a saved session, accepts an image, reports cache usage and subscription quota, and handles cancellation/rate limits without duplicate actions. Validate against the MiniMax dashboard, not estimated subscription token prices.

After implementation, the intended flow is:

```sh
filo --auth minimax
filo --model minimax-m-plan/MiniMax-M3.1-Flash
```

An environment-only path should also work using `MINIMAX_M_PLAN_API_KEY`. These commands are implemented; rebuild Filo before using them.

The subscription is technically compatible with the intended integration. If the purchase is specifically for Filo, complete the offline implementation and fixtures first, then purchase a monthly tier and run the live acceptance checks before making a longer commitment. Go/Explore/Build differ in included usage; the appropriate tier depends on actual workload. Checkout pricing, promotions, account entitlement, and regional access must be confirmed on the user's account. No authenticated request, subscription purchase, or live billing verification was performed for this plan.

The checked reference source is an architectural reference, not the service contract: it retains legacy model cards and some stale options. Current official endpoint documentation plus captured live fixtures should decide conflicts. The shared protocol cleanup is a prerequisite, so MiniMax does not inherit Claude product behavior and future Messages-compatible vendors can use the same neutral foundation.
