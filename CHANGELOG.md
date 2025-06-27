# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.1](https://github.com/pmcgleenon/heavykeeper-rs/compare/v0.5.0...v0.5.1) - 2025-06-27

### Other

- Merge pull request #44 from pmcgleenon/dependabot/cargo/mockall-0.13
- added Send trait to  rng

## [0.5.0](https://github.com/pmcgleenon/heavykeeper-rs/compare/v0.4.0...v0.5.0) - 2025-06-20

### Other

- updated decay logic + added tests
- add increment argument to TopK::add

## [0.4.0](https://github.com/pmcgleenon/heavykeeper-rs/compare/v0.3.1...v0.4.0) - 2025-05-31

### Other

- Merge pull request #37 from pmcgleenon/dependabot/cargo/criterion-0.6.0
- [**breaking**] use reference in add method to avoid cloning

## [0.3.1](https://github.com/pmcgleenon/heavykeeper-rs/compare/v0.3.0...v0.3.1) - 2025-04-12

### Other

- fixed clippy warning for manual_hash_one
- removed unused dependency

## [0.3.0](https://github.com/pmcgleenon/heavykeeper-rs/compare/v0.2.7...v0.3.0) - 2025-03-30

### Added

- try hash composition instead of recalculating new hash
- replaced external priority queue with internal implementation


## [0.2.7](https://github.com/pmcgleenon/heavykeeper-rs/compare/v0.2.6...v0.2.7) - 2025-03-15

### Fixed

- fixed word counting app

### Other

- added script for test data, README tidy-ups

## [0.2.6](https://github.com/pmcgleenon/heavykeeper-rs/compare/v0.2.5...v0.2.6) - 2025-02-11

### Added

- added merge API to merge heavykeeper structs

### Other

- added badges for crate, license, github status

## [0.2.5](https://github.com/pmcgleenon/heavykeeper-rs/compare/v0.2.4...v0.2.5) - 2025-02-06

### Other

- random - dependabot upgrade
- Update rand requirement from 0.8.5 to 0.9.0
- Update rand_distr requirement from 0.4.3 to 0.5.0

## [0.2.4](https://github.com/pmcgleenon/heavykeeper-rs/compare/v0.2.3...v0.2.4) - 2024-11-30

### Other

- Switched to 64 bit counts
- refactored main program
- used mmap for reading file
- modified string processing
- updated count test
