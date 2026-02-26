# uparser

A parser combinator library using a functional approach, sum types and
generics.

## Installation

```bash
# for a library
uv add https://github.com/pcastellazzi/uparser.git
```

## Development

```bash
git clone https://github.com/pcastellazzi/uparser.git
cd uparser
prek install --install-hooks --hook-type pre-push
```

## Tasks

```bash
make all          # shortcut to make install check coverage
make clean        # alias for git clean, *MAY LOOSE PROGRESS* when used on a dirty repo
make check        # run all prek hooks in all files
make coverage     # run the test suite with coverage tracking enabled
make test         # run the test suite
make integration  # run the test suite in multiple python versions
```
