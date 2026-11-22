# SDLC Recommendation

## Recommended SDLC: Spiral

## Alternatives Considered

- **Agile**: Agile's iterative speed conflicts with immutable audit logs (REQ-NF-005) and 99.95% availability (REQ-NF-001) requiring extensive verification phases
- **V-Shaped**: V-Shaped's rigid phase gates struggle with idempotency (REQ-F-005) and rollback requirements (REQ-F-006) that demand continuous validation

## Software Engineering Justification

The requirements demand rigorous security (REQ-F-007, REQ-NF-006), immutable auditability (REQ-NF-005), and strict SLAs (REQ-NF-002, REQ-NF-003) that align with Spiral's risk-driven iterative cycles. Spiral's phased risk analysis matches the need for AML screening (REQ-F-007) and encryption compliance (REQ-NF-006), while its iterative testing accommodates idempotency (REQ-F-005) and rollback (REQ-F-006) requirements. Agile's flexibility would struggle with the 7-year data retention (REQ-NF-005) and 99.95% availability (REQ-NF-001) constraints requiring extensive verification.

## Key Risks and Mitigations

- **Risk:** Conflicting demands between rapid payment iteration (REQ-NF-003) and immutable auditability (REQ-NF-005)
  - **Mitigation:** Implement continuous integration with audit log verification in each Spiral iteration
- **Risk:** AML screening (REQ-F-007) requiring real-time checks against evolving sanctions lists
  - **Mitigation:** Integrate real-time sanctions database synchronization during Spiral risk analysis phases
- **Risk:** Maintaining 99.95% availability (REQ-NF-001) with 7-year data retention (REQ-NF-005)
  - **Mitigation:** Use distributed architecture with geo-redundant storage and automated retention purges in each Spiral cycle
