# Changelog

All notable changes to `casm_calibrator` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased] — 2026-05-17

### Changed (breaking)

- `svd_calibrate()` now requires `data=` as a keyword argument. The
  parameter was previously positional-or-keyword; any caller passing the
  full-triangle visibilities positionally (third argument) must be updated
  to use `data=`. Rationale: SVD requires the full N x N Hermitian matrix,
  which can only come from the full-triangle visibility array, not the
  ref+target subset on `fs`. Making it keyword-only forces explicit intent
  at call sites and prevents the easy mistake of passing `fs` data twice.

### Added

- `SVDConfig.__post_init__` auto-coerces a string `svd_mode` to the
  matching `SVDMode` enum value. Raises `TypeError` (not `ValueError`) for
  unrecognized strings. Allows notebook callers to pass `svd_mode="phase-only"`
  without importing `SVDMode`.

- `save_calibration` gains an `overwrite=False` keyword argument. When the
  target path already exists and `overwrite` is `False` (the default), the
  function raises `FileExistsError` with the path in the message. Pass
  `overwrite=True` to replace an existing file. Parent directories are
  created automatically in both cases.

### Fixed

- `_prepare_svd_input` else-branch comment updated: the comment now reflects
  that both `RAW` and `COMPLEX` modes reach this branch and receive an
  identical raw-copy input matrix. Previously the comment said only `RAW`
  reached the else branch, which was misleading. The underlying behavior is
  unchanged; gain extraction for `COMPLEX` is still correct (it uses
  `U[:,0] * sqrt(sigma[0])` after SVD). A follow-up issue covers whether
  `COMPLEX` should have a distinct input preparation step (see SF-16 in
  `phase3_consolidated_findings.md`).
