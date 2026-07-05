# Contributing

Thanks for taking the time to contribute. This project is a real-world stress-test of LLM serving infrastructure, and any improvements, bug reports, or ideas are welcome.

## How to contribute

### Report a bug

Open an [issue](https://github.com/ZAID646/llm-serving-lab/issues) with:
- What you were doing (quantization, serving, load testing, etc.)
- The exact error message or log output
- Your hardware specs (GPU, VRAM, driver version)
- Docker images and versions used

### Suggest an enhancement

Open an [issue](https://github.com/ZAID646/llm-serving-lab/issues) with the `enhancement` label. Tell us what you want to add and why.

### Submit a pull request

1. Fork the repo
2. Create a branch (`git checkout -b feature/my-change`)
3. Make your changes
4. Test them (see below)
5. Commit and push (`git push origin feature/my-change`)
6. Open a pull request

### What makes a good PR

- One change per PR — don't bundle unrelated fixes
- Clear commit messages explaining why the change was made
- If it changes infrastructure config (Docker, compose, monitoring), mention what you tested
- Update the README if your change affects how someone runs the project

## Code style

- Python: ruff with default config (`pip install ruff && ruff check .`)
- YAML: 2-space indentation, no tabs
- Shell: `shellcheck` clean
- No trailing whitespace
- Keep it readable over clever

## Testing

- **Quantization**: Run the quantize container and verify the model pushes to HF
- **Serving**: `docker compose up -d` and check `curl localhost:8000/health`
- **Load test**: Run a short headless locust test (`-u 10 -r 5 --run-time 30s`)
- **Monitoring**: Check Grafana at `localhost:3000` — dashboard should auto-load

If you're adding a new service or changing the compose wiring, verify all services come up cleanly with `docker compose ps`.

## Project structure

```
docker/quantize/     — AWQ calibration pipeline (separate container)
docker/locust/       — Load testing client
prometheus/          — Prometheus scrape config
grafana/             — Auto-provisioned dashboards and datasources
k3s/                 — Kubernetes manifests
```

Changes to `docker-compose.yml` affect all services — be careful with port mappings and network config.

## Questions

Open a [discussion](https://github.com/ZAID646/llm-serving-lab/discussions) or reach out on GitHub.

PRs are always welcome.
