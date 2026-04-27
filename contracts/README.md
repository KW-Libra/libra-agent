# Contracts

This directory contains versioned JSON Schemas for payloads exchanged by `libra-agent` and `libra-backend`.

`libra-backend` is expected to be implemented in Spring Boot, so these schemas act as the language-neutral contract between Java DTOs and Python runtime models.

Current contracts:

- `judge-run-request.schema.json`
- `judge-run-result.schema.json`
- `evaluation-request.schema.json`
- `evaluation-result.schema.json`
- `push-trigger-event.schema.json`
- `user-approval-response.schema.json`

Keep one schema file per payload and update the versioned contract before changing runtime shape.
