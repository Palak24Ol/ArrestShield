# Container deployment

The image contains code and pinned dependencies, not private data or model weights. Build from the repository root:

```powershell
docker build -f deploy/Dockerfile -t arrestshield-ml:research .
```

Run with the local artifact tree mounted read-only and bind the published port to localhost:

```powershell
docker run --rm -p 127.0.0.1:8000:8000 -v C:\ArrestShield\artifacts:/app/artifacts:ro arrestshield-ml:research
```

The process runs as the unprivileged `arrestshield` user. Offline model flags prevent runtime checkpoint downloads. Audio is processed in ephemeral temporary files. The checked-in policy still marks the detector as research-only and disables honeypot handoff.

Before any non-local deployment, add TLS termination, authentication, request-rate limits, structured privacy-preserving audit logs, malware/content scanning for uploads, model monitoring, and a human-reviewed rollback process. Do not expose this research container directly to the Internet.
