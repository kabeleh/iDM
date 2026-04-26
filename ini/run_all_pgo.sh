#!/usr/bin/env bash

# Run all pgo*.ini files sequentially with CLASS.

set -u
shopt -s nullglob

files=(pgo*.ini)
if [[ ${#files[@]} -eq 0 ]]; then
    echo "No files matched pgo*.ini"
    exit 1
fi

overall_rc=0
for ini in "${files[@]}"; do
    echo "=== Running: $ini ==="
    if ! ./class "$ini"; then
        echo "FAILED: $ini"
        overall_rc=1
    fi
    echo
done

exit "$overall_rc"
