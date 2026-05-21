# CLI Reference

NIRSPY provides two CLI commands:

## `nirspy serve`

Launch the Dash GUI server.

```bash
nirspy serve [--host HOST] [--port PORT] [--debug]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `127.0.0.1` | Host to bind to |
| `--port` | `8050` | Port number |
| `--debug` | off | Enable Dash debug mode |

## `nirspy run`

Execute a saved pipeline from the command line.

```bash
nirspy run PIPELINE_YAML --input SNIRF_FILE --output OUTPUT_DIR [--verbose]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `PIPELINE_YAML` | Yes | Path to the pipeline YAML file |
| `--input` | Yes | Path to the input SNIRF file |
| `--output` | Yes | Output directory for results |
| `--verbose` | No | Enable verbose logging |

### Example

```bash
nirspy run best-practices.yml --input motor_task.snirf --output results/
```
