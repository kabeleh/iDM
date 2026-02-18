# ============================================================================
# Makefile overrides for building CLASS with Clang / AOCC
#
# How to use (from CLASS root):
#   1. Back up or note your current Makefile settings
#   2. Apply the changes below to the Makefile  (the CC, CPP, AR, OPTFLAG lines)
#   3. make clean && make class -j
#   4. Run the benchmark: ./benchmark/run_benchmark.sh clang
#
# Diff against the default GCC settings:
# ============================================================================
#
# --- GCC (original) ---
# CC       = gcc
# CPP      = g++ --std=c++11 -fpermissive -Wno-write-strings
# AR       = gcc-ar rv
# OPTFLAG  = -O3
# OPTFLAG += -funroll-loops -ftree-vectorize -ftree-slp-vectorize -flto=auto -fPIC
# OPTFLAG += -march=native -mtune=native
# OPTFLAG += -fno-math-errno -fno-trapping-math
#
# --- Clang / AOCC ---
# CC       = clang
# CPP      = clang++ --std=c++11 -fpermissive -Wno-write-strings
# AR       = ar rv          # or llvm-ar rv  (needed for full LTO bitcode archives)
# OPTFLAG  = -O3
# OPTFLAG += -funroll-loops -ftree-vectorize -ftree-slp-vectorize -flto=thin -fPIC
# OPTFLAG += -march=native -mtune=native
# OPTFLAG += -fno-math-errno -fno-trapping-math
#
# Key differences:
#   - CC/CPP: gcc/g++ → clang/clang++
#   - AR:     gcc-ar  → ar  (or llvm-ar for LTO)
#   - LTO:    -flto=auto (GCC) → -flto=thin (Clang)
#     Note: -flto=auto is not recognized by Clang.
#     -flto=thin is Clang's incremental LTO (faster link, similar perf).
#     -flto=full is also available if you want maximal cross-TU optimisation.
#
# Everything else (OMPFLAG, CCFLAG, LDFLAG, etc.) is compatible as-is.
# If you use PGO, the Clang flags differ:
#   GCC:   -fprofile-generate / -fprofile-use
#   Clang: -fprofile-instr-generate / -fprofile-instr-use=<merged.profdata>
#   (You also need to merge raw profiles with llvm-profdata before the -use step.)
# ============================================================================
