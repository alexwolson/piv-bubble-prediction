#!/bin/bash
# Transfer Zarr archive to nibi cluster using rsync
#
# Usage: bash scripts/transfer/sync_data.sh [SOURCE] [DEST] [OPTIONS]
#
# Arguments:
#   SOURCE: Local path to Zarr archive (default: data/raw/all_experiments.zarr/)
#   DEST: Destination path on cluster (default: /scratch/$NIBI_USER/data/raw/)
#
# Environment variables:
#   NIBI_USER: Username on nibi cluster (default: awolson)
#   NIBI_HOST: Hostname for nibi (default: nibi.alliancecan.ca)
#
# Examples:
#   # Transfer to scratch space (default)
#   bash scripts/transfer/sync_data.sh
#
#   # Custom source and destination
#   bash scripts/transfer/sync_data.sh ./my_data.zarr/ awolson@nibi.alliancecan.ca:/scratch/awolson/data/

set -e  # Exit on error

# Default values
SOURCE=${1:-data/raw/all_experiments.zarr/}
NIBI_USER=${NIBI_USER:-awolson}
NIBI_HOST=${NIBI_HOST:-nibi.alliancecan.ca}

# Determine destination path
if [ -n "$2" ]; then
    DEST="$2"
else
    # Use scratch space (default location for cluster data)
    DEST="$NIBI_USER@$NIBI_HOST:/scratch/$NIBI_USER/data/raw/"
fi

# Validate source exists
if [ ! -d "$SOURCE" ]; then
    echo "Error: Source directory not found: $SOURCE"
    exit 1
fi

echo "Transferring Zarr archive to nibi cluster..."
echo "Source: $SOURCE"
echo "Destination: $DEST"
echo ""

# Check if rsync is available
if ! command -v rsync &> /dev/null; then
    echo "Error: rsync not found. Install rsync to use this script."
    exit 1
fi

# Rsync options:
# -a: archive mode (preserves permissions, timestamps, etc.)
# -v: verbose
# -z: compress during transfer
# -h: human-readable output
# --progress: show progress
# --partial: keep partial files if transfer is interrupted
# --partial-dir: directory for partial files
# --delete: delete files in destination that don't exist in source (use with caution!)
# --exclude: exclude patterns (add patterns for temporary files if needed)

RSYNC_OPTS=(
    -avz
    -h
    --progress
    --partial
    --partial-dir=.rsync-partial
)

# Add exclude patterns for temporary files if needed
# RSYNC_OPTS+=(--exclude="*.tmp" --exclude="*.lock")

# Perform transfer
echo "Starting transfer..."
echo "Note: First transfer may take a while depending on data size."
echo "Subsequent transfers will only sync changes (incremental)."
echo ""

rsync "${RSYNC_OPTS[@]}" "$SOURCE" "$DEST" || {
    echo ""
    echo "Transfer failed. Check:"
    echo "1. SSH access to nibi: ssh $NIBI_USER@$NIBI_HOST"
    echo "2. Multifactor authentication (MFA) is configured for your account"
    echo "3. Destination directory exists: ssh $NIBI_USER@$NIBI_HOST 'mkdir -p /scratch/$NIBI_USER/data/raw'"
    echo "4. Network connectivity"
    exit 1
}

echo ""
echo "Transfer completed successfully!"
echo ""
echo "To verify on nibi, run:"
echo "  ssh $NIBI_USER@$NIBI_HOST 'ls -lh /scratch/$NIBI_USER/data/raw/'"
