#!/bin/bash

ROOT="/data/ascher02/uqmmune1/Splice2/GWAS2m"

export PIPELINE_ROOT="$ROOT"

source "$ROOT/resource_paths.env"

echo "Main environment:"
echo "  $ROOT/envs/pipeline"

echo
echo "Activate with:"
echo
echo "  conda activate $ROOT/envs/pipeline"

echo
echo "SpliceAI:"
echo "  $ROOT/envs/spliceai/bin/spliceai"

echo
echo "Pangolin:"
echo "  $ROOT/envs/pangolin/bin/pangolin"
