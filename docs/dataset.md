# Dataset Guide

The default dataset abstraction is a signer-aware synthetic dataset built for reproducible validation.

- each sample includes a label and signer identifier
- splits avoid data leakage by assigning signers to one partition only
- landmark arrays are modeled as time x feature sequences
- the format is designed to support real landmark extraction later without changing the interface

The project is intentionally generic so it can accept public or custom sign datasets with a consistent schema.
