#!/usr/bin/env python3
"""
apply_hsbm_patches.py -- surgical, verified, idempotent patches to the HSBM sources.

Run from the repository root:

    python3 apply_hsbm_patches.py            # apply
    python3 apply_hsbm_patches.py --check    # report only, change nothing
    python3 apply_hsbm_patches.py --revert   # restore from .orig backups

Every edit is an exact-string replacement.  If the expected text is not found
the script aborts before writing anything, so a partial patch is impossible.
Backups are written once to <file>.orig.

What it does
------------
utils.hpp           : portable <omp.h> include (+ no-op shims when built w/o OpenMP)
blockallocator.hpp  : replace the *global named* critical section with a per-pool
                      lock drawn from a hashed lock table.  A named critical is
                      shared by every BlockMemoryPool in the process, so thread
                      private scratch pools serialise against each other.
blocksparsematrix.hpp,
strassen.hpp        : add if(!omp_in_parallel()) to every `omp parallel for`, so
                      the routines stay usable standalone but cost nothing when
                      called from inside an outer parallel region.
strassen.hpp        : - correct the scratch-pool bound (was O(h^2 (7/4)^depth),
                        the depth-first live set is 4h^2)
                      - size the pointer table independently (it needs ~24h^2)
                      - optional explicit block extents on the dry run and on
                        hybrid_recurse, so a sub-grid can be multiplied without
                        copying (all new arguments are defaulted: existing call
                        sites behave exactly as before)
                      - drop the square-operand asserts; derive max_level from
                        min(nib,njb,nkb)
                      - dry-run progress printf behind -DHSBM_DRYRUN_VERBOSE

The cost model and the Strassen_fac penalties are NOT touched.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# (file, description, old, new)
# ----------------------------------------------------------------------------
PATCHES = []


def P(fn, desc, old, new, count=1, sentinel=None, regex=False):
    """sentinel: text whose presence means the patch is already applied."""
    PATCHES.append((fn, desc, old, new, count, sentinel, regex))


# --- utils.hpp : omp shim -----------------------------------------------------
P("utils.hpp", "portable omp.h include",
  """#ifndef UTILS_HPP
#define UTILS_HPP
#include <unistd.h>
""",
  """#ifndef UTILS_HPP
#define UTILS_HPP
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#else
static inline int  omp_in_parallel()    { return 0; }
static inline int  omp_get_thread_num() { return 0; }
static inline int  omp_get_max_threads(){ return 1; }
#endif
""", sentinel="omp_get_max_threads(){ return 1; }")

# --- blockallocator.hpp : per-pool lock --------------------------------------
P("blockallocator.hpp", "per-pool lock table (replaces global named critical)",
  """#include "blockallocator.h"
#include <cassert>
""",
  """#include "blockallocator.h"
#include <cassert>
#include <cstdint>
#include "utils.hpp"

// A `#pragma omp critical (blockallocator)` is keyed by *name*, so every
// BlockMemoryPool in the process shares one global lock.  With per-thread
// scratch pools that serialises the whole recursion.  These guards hash the
// pool address into a table of independent locks instead, so a thread-private
// pool takes an uncontended lock (~20ns) rather than a contended global one.
// Define HSBM_POOL_NO_LOCK only if you can guarantee every pool is touched by
// exactly one thread (the BSM baseline cannot -- it allocates output blocks
// lazily from a shared pool inside a parallel region).
#ifdef HSBM_POOL_NO_LOCK
struct _hsbm_pool_guard { explicit _hsbm_pool_guard(const void*) {} };
#else
class _hsbm_pool_guard final {
  public:
    explicit _hsbm_pool_guard(const void* key)
      : _l(_table() + ((((uintptr_t)key) >> 5) % _NLOCK)) { omp_set_lock(_l); }
    ~_hsbm_pool_guard() { omp_unset_lock(_l); }
    _hsbm_pool_guard(const _hsbm_pool_guard&) = delete;
    _hsbm_pool_guard& operator=(const _hsbm_pool_guard&) = delete;
  private:
    static constexpr uintptr_t _NLOCK = 251;
    static omp_lock_t* _table(){
      static omp_lock_t* t = [](){
        omp_lock_t* v = new omp_lock_t[_NLOCK];
        for(uintptr_t i=0;i<_NLOCK;++i) omp_init_lock(&v[i]);
        return v;
      }();
      return t;
    }
    omp_lock_t* _l;
};
#endif
""", sentinel="_hsbm_pool_guard")

P("blockallocator.hpp", "swap critical for pool guard (allocate + deallocate)",
  """  #pragma omp critical (blockallocator)
  {
""",
  """  {
    const _hsbm_pool_guard _hsbm_g(this);
""", count=2, sentinel="_hsbm_pool_guard _hsbm_g(this)")

# --- omp parallel for guards --------------------------------------------------
for _f in ("blocksparsematrix.hpp", "strassen.hpp"):
    P(_f, "guard omp parallel for with if(!omp_in_parallel())",
      r"\#pragma omp parallel for(?! if\(!omp_in_parallel\(\)\))",
      "#pragma omp parallel for if(!omp_in_parallel())",
      count=-1, regex=True)  # -1 == replace every occurrence

# --- strassen.hpp : scratch bound --------------------------------------------
P("strassen.hpp", "correct scratch-pool bound",
  """inline size_t compute_total_blocks(size_t h, size_t depth)
{
    size_t total=0, grid_h=h, n_calls=1;
    for(size_t l=0;l<depth;++l){
        total  += (size_t)13 * n_calls * grid_h * grid_h;
        grid_h  = (grid_h > 1) ? grid_h/2 : 1;
        n_calls *= 7;
    }
    return total;
}""",
  """inline size_t compute_total_blocks(size_t h, size_t depth)
{
    // The recursion is depth-first: at most one child is live at a time, so the
    // old sum 13 * 7^l * (h/2^l)^2 over-counts by a factor (7/4)^depth (about
    // 30x at depth 6).  Live set, with A(g) the peak inside
    // strassen_matmult_acc on g x g grids and R(g) the peak inside
    // strassen_recurse:
    //     A(g) = g^2 + R(g/2),  R(g) = 8g^2 + A(g)  =>  R(g) = 9g^2 + R(g/2)
    //     R(g) = 9g^2 * 4/3 = 12 g^2
    // The caller adds 4h^2 for the output grids, giving 16h^2 in total; this
    // function returns the 4h^2 part so the call site is unchanged.
    (void)depth;
    return 4lu*h*h;
}""")

# --- strassen.hpp : hybrid_recurse explicit extents ---------------------------
P("strassen.hpp", "hybrid_recurse: optional explicit block extents",
  """                    PtrBumpAlloc<Num>& ptr_alloc)
{
    const size_t nib = A_mat.nrowblocks();
    const size_t njb = B_mat.ncolblocks();
    const size_t nkb = A_mat.ncolblocks();
""",
  """                    PtrBumpAlloc<Num>& ptr_alloc,
                    size_t nib_in = 0, size_t njb_in = 0, size_t nkb_in = 0)
{
    // Explicit extents let a caller multiply a leading sub-grid of the operands
    // (the "perfectly sized" core) without copying anything.  Zero means "use
    // the whole matrix", which is what every pre-existing call site does.
    const size_t nib = nib_in ? nib_in : A_mat.nrowblocks();
    const size_t njb = njb_in ? njb_in : B_mat.ncolblocks();
    const size_t nkb = nkb_in ? nkb_in : A_mat.ncolblocks();
""")

P("strassen.hpp", "hybrid_recurse: forward extents on descent",
  """                                 block_nr, block_nc,
                                 scratch_alloc, ptr_alloc);""",
  """                                 block_nr, block_nc,
                                 scratch_alloc, ptr_alloc,
                                 nib, njb, nkb);""")

# --- strassen.hpp : dry run signature ----------------------------------------
P("strassen.hpp", "dry run: extents, level cap, cost export",
  """std::vector<std::vector<bool>> matmult_strassen_dryrun(
             const BlockSparseMatrix<Num>& A_mat, 
             const BlockSparseMatrix<Num>& B_mat, 
             const Num thresh_per_block)
{""",
  """std::vector<std::vector<bool>> matmult_strassen_dryrun(
             const BlockSparseMatrix<Num>& A_mat, 
             const BlockSparseMatrix<Num>& B_mat, 
             const Num thresh_per_block,
             const size_t nib_in = 0, const size_t njb_in = 0, const size_t nkb_in = 0,
             const size_t max_level_cap = 0,
             std::vector<size_t>* top_flops_out = nullptr)
{""")

P("strassen.hpp", "dry run: rectangular extents, max_level from min axis",
  """  const size_t nib = integer_division_round_up(ni,i_block_size);
  assert(nib == transA? A_mat.ncolblocks() : A_mat.nrowblocks());
  const size_t njb = integer_division_round_up(nj,j_block_size);
  assert(njb == transB? B_mat.nrowblocks() : B_mat.ncolblocks());
  const size_t nkb = integer_division_round_up(nk,k_block_size);
  assert(nkb == transA? A_mat.nrowblocks() : A_mat.ncolblocks());
  assert(nkb == transB? B_mat.ncolblocks() : B_mat.nrowblocks());

  //only quadratic so far
  assert(ni == nj);
  assert(ni == nk);
  assert(nib == njb);
  assert(nib == nkb);

  const size_t max_level = (size_t)std::log2((double)nib+0.5);""",
  """  // Strassen-Winograd needs the three extents to be *even*, not equal, so the
  // square restriction is lifted here: the recursion depth is set by the
  // shortest axis and ragged super-blocks keep falling onto the sparse path via
  // is_edge_case below, exactly as they already did for odd square grids.
  // (The asserts that stood here were also no-ops: `assert(nib == transA ? x : y)`
  // parses as `assert((nib == transA) ? x : y)`.)
  const size_t nib = nib_in ? nib_in : integer_division_round_up(ni,i_block_size);
  const size_t njb = njb_in ? njb_in : integer_division_round_up(nj,j_block_size);
  const size_t nkb = nkb_in ? nkb_in : integer_division_round_up(nk,k_block_size);

  size_t max_level = (size_t)std::log2((double)std::min(std::min(nib,njb),nkb)+0.5);
  if(max_level_cap != 0lu && max_level > max_level_cap) max_level = max_level_cap;""")

P("strassen.hpp", "dry run: silence per-level printf",
  """    const size_t n_flops_total = std::accumulate(nflops_per_level[level+1].cbegin(),nflops_per_level[level+1].cend(),0lu);
    printf("total flops on level %lu: %lu\\n",level+1, n_flops_total);
  }
  return do_Strasssen_decision_tree;""",
  """#ifdef HSBM_DRYRUN_VERBOSE
    const size_t n_flops_total = std::accumulate(nflops_per_level[level+1].cbegin(),nflops_per_level[level+1].cend(),0lu);
    printf("total flops on level %lu: %lu\\n",level+1, n_flops_total);
#endif
  }
  // Export the top-level per-super-block cost estimate.  The parallel driver
  // uses it to run the largest tiles first (longest-processing-time-first),
  // which is what keeps a dynamic schedule balanced.
  if(top_flops_out != nullptr) *top_flops_out = nflops_per_level[max_level];
  return do_Strasssen_decision_tree;""")

# --- strassen.hpp : entry point ----------------------------------------------
P("strassen.hpp", "entry point: drop square assert",
  """    assert(nib == njb && nib == nkb);""",
  """    // (square operands no longer required)""")

P("strassen.hpp", "entry point: size the pointer table independently",
  """    std::vector<const Mat<Num>*> ptr_buf(pool_blocks);
    PtrBumpAlloc<Num> ptr_alloc{ ptr_buf.data(), pool_blocks, 0 };""",
  """    // The pointer table is NOT the same size as the block pool: quadrant views
    // and grid pointer tables peak around 24h^2 slots while the memory pool
    // peaks at 16h^2 blocks.  Pointers are 8 bytes, so be generous here.
    const size_t ptr_slots = 40lu*h_top*h_top + 128lu;
    std::vector<const Mat<Num>*> ptr_buf(ptr_slots);
    PtrBumpAlloc<Num> ptr_alloc{ ptr_buf.data(), ptr_slots, 0 };""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only")
    ap.add_argument("--revert", action="store_true", help="restore .orig backups")
    ap.add_argument("--root", default=".", help="repository root")
    args = ap.parse_args()
    root = Path(args.root)

    files = sorted({p[0] for p in PATCHES})

    if args.revert:
        for f in files:
            b = root / (f + ".orig")
            if b.exists():
                shutil.copy2(b, root / f)
                print(f"reverted {f}")
            else:
                print(f"no backup for {f}, skipped")
        return 0

    for f in files:
        if not (root / f).exists():
            print(f"ERROR: {f} not found in {root.resolve()}", file=sys.stderr)
            return 1

    texts = {f: (root / f).read_text() for f in files}
    applied, already = [], []

    for fn, desc, old, new, count, sentinel, regex in PATCHES:
        t = texts[fn]
        if sentinel is not None and sentinel in t:
            already.append(f"{fn}: {desc}")
            continue
        if regex:
            t2, n = re.subn(old, new.replace("\\", "\\\\"), t)
            if n == 0:
                already.append(f"{fn}: {desc}")
                continue
            texts[fn] = t2
            applied.append(f"{fn}: {desc} ({n}x)")
            continue
        if old not in t:
            if new in t:
                already.append(f"{fn}: {desc}")
                continue
            print(f"ERROR: {fn}: cannot find anchor for '{desc}'.", file=sys.stderr)
            print("       The file differs from the version this script was "
                  "written against; nothing has been written.", file=sys.stderr)
            print(f"       Missing text:\n{old[:400]}", file=sys.stderr)
            return 1
        n = t.count(old)
        if count > 0 and n != count:
            print(f"ERROR: {fn}: expected {count} occurrence(s) for '{desc}', "
                  f"found {n}; nothing written.", file=sys.stderr)
            return 1
        texts[fn] = t.replace(old, new)
        applied.append(f"{fn}: {desc} ({n}x)")

    for a in already:
        print(f"  already applied  {a}")
    for a in applied:
        print(f"  apply            {a}")

    if args.check:
        print("\n--check: nothing written.")
        return 0
    if not applied:
        print("\nnothing to do.")
        return 0

    for f in files:
        b = root / (f + ".orig")
        if not b.exists():
            shutil.copy2(root / f, b)
        (root / f).write_text(texts[f])
    print(f"\nwrote {len(files)} file(s); backups in *.orig")
    return 0


if __name__ == "__main__":
    sys.exit(main())
