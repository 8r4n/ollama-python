"""
Airgapped model transfer example
=================================

This script demonstrates how to:
  1. Export a model to a portable .tar.gz archive on a network-connected machine.
  2. Import that archive into a locally-running Ollama instance on an airgapped
     (network-disconnected) machine.

Typical workflow
----------------
  Connected machine:
    python airgap-transfer.py export llama3.2:latest llama3.2-latest.tar.gz

  Transfer the archive to the airgapped machine via USB drive, SCP, or any
  out-of-band mechanism available in your environment.

  Airgapped machine (Ollama must be running locally):
    python airgap-transfer.py import llama3.2:latest llama3.2-latest.tar.gz

Usage
-----
  python airgap-transfer.py export <model> <archive.tar.gz>
  python airgap-transfer.py import <model> <archive.tar.gz>

Options
-------
  --models-dir PATH   Override the Ollama models directory.
                      Defaults to OLLAMA_MODELS env var or ~/.ollama/models.
"""

import argparse
import sys

import ollama


def cmd_export(model: str, archive: str, models_dir: str | None) -> None:
  print(f'Exporting {model!r} -> {archive!r} ...')
  ollama.save(model, archive, models_dir=models_dir)
  print('Export complete.')
  print()
  print('Transfer the archive to the airgapped machine, then run:')
  print(f'  python airgap-transfer.py import {model} {archive}')


def cmd_import(model: str, archive: str, models_dir: str | None) -> None:
  print(f'Importing {archive!r} -> {model!r} ...')
  ollama.load(model, archive, models_dir=models_dir)
  print('Import complete.')
  print()
  print('Verifying model is listed ...')
  models = ollama.list()
  names = [m.model for m in models.models]
  # Normalise the requested name to a "name:tag" form for comparison.
  ref = model if ':' in model else f'{model}:latest'
  if any(n and (n == ref or n.startswith(ref.split(':')[0])) for n in names):
    print(f'  {model!r} is available.')
  else:
    print(f'  {model!r} not found in list — Ollama may need a moment to index it.')
  print()
  print('Run a quick inference test:')
  print(f'  import ollama; print(ollama.generate(model={model!r}, prompt="Hello").response)')


def main() -> None:
  parser = argparse.ArgumentParser(description='Export/import Ollama models for airgapped transfer.')
  parser.add_argument('command', choices=['export', 'import'], help='export or import')
  parser.add_argument('model', help='Model name, e.g. llama3.2:latest')
  parser.add_argument('archive', help='Path to the .tar.gz archive')
  parser.add_argument('--models-dir', default=None, help='Override the Ollama models directory')
  args = parser.parse_args()

  try:
    if args.command == 'export':
      cmd_export(args.model, args.archive, args.models_dir)
    else:
      cmd_import(args.model, args.archive, args.models_dir)
  except FileNotFoundError as exc:
    print(f'Error: {exc}', file=sys.stderr)
    sys.exit(1)
  except ValueError as exc:
    print(f'Error: {exc}', file=sys.stderr)
    sys.exit(1)


if __name__ == '__main__':
  main()
