# eBay Live Web: CI/CD Recovery Strategy

A 10% test pass rate means CI is currently decoration. Nobody gates on it, nobody trusts it, and every real regression it catches is drowned out by 90 failures people have learned to ignore. The goal of this strategy is a pipeline that is green by default, where a red build is rare, meaningful, and blocking.

The path there is quarantine first, ratchet second, deploy automation last. Trying to fix 90% of a suite before turning on enforcement takes months and dies of fatigue. Turning on enforcement against the 10% that passes takes a week and starts compounding immediately.

## Assumptions

This doc was written without direct access to the repo. It assumes a web application with a mix of unit, integration, and browser/e2e tests, deployed as a live-streaming commerce surface where availability during live events is the dominant risk. Where the actual stack differs (CI provider, test runner, deploy target), the mechanics below translate; the sequencing does not change.

## Phase 0: Triage the 90% (week 1)

Run the full suite three times and diff the results. Every failing test lands in exactly one bucket:

- **Broken product code.** The test is right and the app is wrong. These are latent bugs; file them as such.
- **Rotted tests.** The app changed, the test didn't. Delete or rewrite; a test asserting last year's behavior has negative value.
- **Flaky.** Passes on some runs, fails on others. Timing, shared state, network, test-order dependence.
- **Environmental.** Fails in CI, passes locally (or vice versa): missing services, credentials, browser versions, resource limits.

The three-run diff separates flaky from consistently-failing mechanically, without arguing about it test by test. Expect flaky and rotted to dominate; in suites this degraded, genuine product bugs are usually the smallest bucket.

## Phase 1: Green baseline (week 1–2)

Quarantine every consistently-failing and flaky test with an explicit annotation (`test.fixme`, `@pytest.mark.quarantine`, skip-list file — whatever the runner supports). The annotation must carry a ticket link. A quarantined test still runs in a separate non-blocking job so its status stays visible; it just can't fail the build.

Then, immediately:

1. Make the remaining suite a **required check** on the main branch. Branch protection on, force-push off, no merges on red.
2. Add lint and typecheck as required checks alongside it.
3. Set a hard budget for the blocking suite: under 10 minutes. Anything slower moves to a post-merge or nightly job.

This is the single highest-leverage move in the whole plan. From this point forward the suite only degrades if someone deliberately quarantines a new test, and that action is visible in review.

## Phase 2: Ratchet the quarantine down (weeks 3–10)

The quarantine list is now the backlog, with a number attached. The rules:

- **The list only shrinks.** Adding a test to quarantine requires a ticket and a reviewer. CI fails if the list grows without one.
- **Ownership by area.** Split the list by feature area and assign each slice to the team that owns the code. Unowned tests get deleted after a stated grace period — an unowned failing test is already dead, this just makes it official.
- **Fixed weekly capacity.** Each team commits a small, constant allocation (one engineer-day a week is enough) rather than a heroic sprint. Burn-down is boring on purpose.
- **Delete freely.** A rotted test that nobody can explain gets deleted, not lovingly restored. Coverage that has to be rebuilt should be rebuilt against current behavior.

Publish the quarantine count weekly. The number falling is the progress report; no other status theater is needed.

## Pipeline shape

Three tiers, by feedback speed:

| Tier | Trigger | Contents | Budget | Blocking |
|------|---------|----------|--------|----------|
| PR checks | every push | lint, typecheck, unit + fast integration | < 10 min | yes |
| Main | merge to main | full non-quarantined suite, build artifact | < 30 min | blocks deploy |
| Nightly | schedule | e2e/browser suite, quarantined tests, long soaks | none | no |

For flakes that survive triage: one automatic retry, and every retry-that-passed gets logged and reported. A test that needs its retry more than a few times in a week gets auto-quarantined with a ticket. Retries without reporting hide flakiness; retries with reporting are a detection mechanism.

If merge volume is high enough that main breaks from semantic conflicts between concurrently-green PRs, add a merge queue (GitHub merge queue, Mergify). Defer this until it's an observed problem.

## Deployment

Deploys ride on main being green, and nothing else.

- **Trunk-based flow.** Short-lived branches into main; main is always deployable. Long-lived release branches recreate the integration pain this plan exists to remove.
- **Auto-deploy to staging** on every green main build.
- **Canary to production.** Small traffic slice first, promote on health metrics (error rate, latency, checkout success), auto-rollback on regression. For a live-commerce surface this matters more than test coverage does: the tests catch what someone thought to write down, the canary catches everything else.
- **Feature flags for risk.** Anything touching the live-event path ships dark and gets enabled outside peak hours. Deploy and release become separate decisions.
- **Deploy freeze around major live events**, with a documented break-glass path. Boring, and correct for this domain.

Rollback must be one command and under five minutes. Until that's true, every other safety mechanism is weaker than it looks.

## Culture rules that make it stick

- **Revert first, debug second.** If main goes red, the offending change is reverted within 15 minutes. Debugging happens on a branch. Nobody is fixing forward on a red main.
- **Red blocks everyone, so red is everyone's problem.** The person whose change broke main owns the revert; the on-call for CI owns making sure it happens.
- **New code needs tests to merge.** Enforced socially in review, optionally by a coverage-ratchet on changed lines. A global coverage percentage target is not worth setting; it produces assertion-free tests.

## Metrics

Pass rate stops being interesting once the gate exists (it will read ~100% by construction). Track instead:

- Quarantine count (the burn-down number)
- Flake rate: retries-that-passed per week
- Time-to-green: red-main duration per incident
- Deploy frequency and rollback rate

## Sequencing summary

| Weeks | Work | Exit criterion |
|-------|------|----------------|
| 1 | Triage: 3× run, bucket every failure | Every failing test classified and ticketed |
| 1–2 | Quarantine, required checks on, lint/typecheck gates | Main is green and protected |
| 2–3 | Pipeline tiers, retry-with-reporting, staging auto-deploy | PR feedback < 10 min |
| 3–10 | Quarantine burn-down at fixed weekly capacity | List < 10% of original |
| 4+ | Canary + auto-rollback, flags on live-event paths | One-command rollback, verified |

## Open decisions

- CI provider and runner budget (parallelism is the main lever on the 10-minute PR budget).
- e2e tooling: if the browser suite is the flakiest layer (it usually is), it may be worth rebuilding small in Playwright rather than stabilizing a large legacy suite.
- Merge queue: adopt only when main breakage from concurrent merges is observed.
- Grace period before deleting unowned quarantined tests (two weeks is a reasonable default).
