#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${1:-data/raw}"
PROCESSED_DIR="${2:-data/processed}"
GEOMETRY_DIR="${3:-data/geometry}"
JOBS="${4:-4}"

mkdir -p "$PROCESSED_DIR" "$GEOMETRY_DIR"

for cmd in lronac2isis spiceinit phocube parallel; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: '$cmd' not found in PATH" >&2
        exit 1
    fi
done

[[ -n "${ISISROOT:-}" ]] || {
    echo "ERROR: ISISROOT is not set." >&2
    exit 1
}

[[ -n "${ISISDATA:-}" ]] || {
    echo "ERROR: ISISDATA is not set." >&2
    exit 1
}

echo "ISISROOT : $ISISROOT"
echo "ISISDATA : $ISISDATA"
echo "INPUT    : $INPUT_DIR"
echo "PROCESSED: $PROCESSED_DIR"
echo "GEOMETRY : $GEOMETRY_DIR"
echo "JOBS     : $JOBS"

export PROCESSED_DIR GEOMETRY_DIR

find "$INPUT_DIR" -maxdepth 1 -type f -iname '*.img' -print0 |
parallel \
    --null \
    --jobs "$JOBS" \
    --bar \
    --joblog "$GEOMETRY_DIR/phocube.joblog" \
    --halt soon,fail=1 \
'
    set -euo pipefail

    input="{}"
    filename="$(basename "$input")"
    stem="${filename%.*}"

    raw="$PROCESSED_DIR/${stem}.raw.cub"
    spice="$PROCESSED_DIR/${stem}.spice.cub"
    geometry="$GEOMETRY_DIR/${stem}.geometry.cub"

    # Final output exists: this image is already processed.
    if [[ -f "$geometry" ]]; then
        echo "SKIP: $filename (geometry output exists)"
        exit 0
    fi

    echo
    echo "========== $filename =========="

    # 1. IMG -> raw ISIS cube
    if [[ ! -f "$raw" ]]; then
        echo "[1/3] lronac2isis: $filename"

        lronac2isis \
            from="$input" \
            to="$raw"
    else
        echo "[1/3] SKIP existing: $raw"
    fi

    # 2. Always recreate SPICE cube using the ELLIPSOID shape model.
    echo "[2/3] spiceinit (ELLIPSOID): $raw"

    rm -f "$spice"
    cp "$raw" "$spice"

    spiceinit \
        from="$spice" \
        SHAPE=ELLIPSOID \
        WEB=true

    # 3. Generate phase, emission, incidence bands.
    echo "[3/3] phocube: $spice"

    phocube \
        from="$spice" \
        to="$geometry" \
        phase=true \
        emission=true \
        incidence=true

    echo "DONE: $filename"
'
