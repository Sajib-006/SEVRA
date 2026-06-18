# Security Policy

Do not report exposed credentials or sensitive prompt data in a public issue. Contact the repository
owner privately through their GitHub profile and include the affected file or revision. Revoke any
credential before reporting it.

SEVRA executes an application-provided verifier. Treat model output as untrusted data: validate tool
arguments, sandbox code execution, and apply the same authorization rules used for ordinary requests.
