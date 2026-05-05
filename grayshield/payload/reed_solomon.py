from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


RS_N = 255
RS_K = 127
RS_NSYM = RS_N - RS_K
GF_PRIMITIVE_POLY = 0x11D
GF_GENERATOR = 2


class ReedSolomonDecodeError(Exception):
    """Raised when an RS codeword cannot be decoded reliably."""


@dataclass(frozen=True)
class RSChunkedReport:
    blocks_total: int
    blocks_decoded: int
    blocks_failed: int
    blocks_truncated: int
    corrected_symbol_errors: int
    padding_bytes: int
    dropped_tail_bits: int = 0


def _init_tables() -> Tuple[List[int], List[int]]:
    exp = [0] * 512
    log = [0] * 256
    x = 1
    for i in range(255):
        exp[i] = x
        log[x] = i
        x <<= 1
        if x & 0x100:
            x ^= GF_PRIMITIVE_POLY
    for i in range(255, 512):
        exp[i] = exp[i - 255]
    return exp, log


GF_EXP, GF_LOG = _init_tables()


def gf_add(x: int, y: int) -> int:
    return x ^ y


def gf_sub(x: int, y: int) -> int:
    return x ^ y


def gf_mul(x: int, y: int) -> int:
    if x == 0 or y == 0:
        return 0
    return GF_EXP[GF_LOG[x] + GF_LOG[y]]


def gf_div(x: int, y: int) -> int:
    if y == 0:
        raise ZeroDivisionError("division by zero in GF(256)")
    if x == 0:
        return 0
    return GF_EXP[(GF_LOG[x] + 255 - GF_LOG[y]) % 255]


def gf_pow(x: int, power: int) -> int:
    if power == 0:
        return 1
    if x == 0:
        return 0
    return GF_EXP[(GF_LOG[x] * power) % 255]


def gf_inverse(x: int) -> int:
    if x == 0:
        raise ZeroDivisionError("inverse of zero in GF(256)")
    return GF_EXP[255 - GF_LOG[x]]


def gf_poly_scale(poly: Sequence[int], x: int) -> List[int]:
    return [gf_mul(coeff, x) for coeff in poly]


def gf_poly_add(p: Sequence[int], q: Sequence[int]) -> List[int]:
    length = max(len(p), len(q))
    out = [0] * length
    for i, coeff in enumerate(p):
        out[i + length - len(p)] ^= coeff
    for i, coeff in enumerate(q):
        out[i + length - len(q)] ^= coeff
    return out


def gf_poly_mul(p: Sequence[int], q: Sequence[int]) -> List[int]:
    out = [0] * (len(p) + len(q) - 1)
    for j, qj in enumerate(q):
        if qj == 0:
            continue
        for i, pi in enumerate(p):
            if pi != 0:
                out[i + j] ^= gf_mul(pi, qj)
    return out


def gf_poly_div(dividend: Sequence[int], divisor: Sequence[int]) -> Tuple[List[int], List[int]]:
    msg_out = list(dividend)
    for i in range(len(dividend) - len(divisor) + 1):
        coef = msg_out[i]
        if coef == 0:
            continue
        for j in range(1, len(divisor)):
            if divisor[j] != 0:
                msg_out[i + j] ^= gf_mul(divisor[j], coef)
    split = -(len(divisor) - 1)
    if split == 0:
        return msg_out, []
    return msg_out[:split], msg_out[split:]


def gf_poly_eval(poly: Sequence[int], x: int) -> int:
    y = poly[0]
    for coeff in poly[1:]:
        y = gf_mul(y, x) ^ coeff
    return y


def gf_matrix_solve(matrix: Sequence[Sequence[int]], rhs: Sequence[int]) -> List[int]:
    if len(matrix) != len(rhs):
        raise ValueError("matrix and rhs dimension mismatch")
    if not matrix:
        return []

    n = len(rhs)
    aug = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]

    for col in range(n):
        pivot = None
        for row in range(col, n):
            if aug[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            raise ReedSolomonDecodeError("singular Reed-Solomon linear system")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        inv = gf_inverse(aug[col][col])
        for j in range(col, n + 1):
            aug[col][j] = gf_mul(aug[col][j], inv)

        for row in range(n):
            if row == col or aug[row][col] == 0:
                continue
            factor = aug[row][col]
            for j in range(col, n + 1):
                aug[row][j] ^= gf_mul(factor, aug[col][j])

    return [aug[i][n] for i in range(n)]


def rs_generator_poly(nsym: int) -> List[int]:
    g = [1]
    for i in range(nsym):
        g = gf_poly_mul(g, [1, gf_pow(GF_GENERATOR, i)])
    return g


GENERATOR_POLY = rs_generator_poly(RS_NSYM)


def rs_encode_block(msg: Sequence[int]) -> List[int]:
    if len(msg) != RS_K:
        raise ValueError(f"RS(255,127) encode expects {RS_K} bytes, got {len(msg)}")

    msg_out = list(msg) + [0] * RS_NSYM
    for i in range(RS_K):
        coef = msg_out[i]
        if coef == 0:
            continue
        for j in range(1, len(GENERATOR_POLY)):
            msg_out[i + j] ^= gf_mul(GENERATOR_POLY[j], coef)
    return list(msg) + msg_out[RS_K:]


def rs_calc_syndromes(codeword: Sequence[int], nsym: int = RS_NSYM) -> List[int]:
    return [0] + [gf_poly_eval(codeword, gf_pow(GF_GENERATOR, i)) for i in range(nsym)]


def rs_find_error_locator(synd: Sequence[int], nsym: int) -> List[int]:
    err_loc = [1]
    old_loc = [1]

    for i in range(nsym):
        delta = synd[i + 1]
        for j in range(1, len(err_loc)):
            delta ^= gf_mul(err_loc[-(j + 1)], synd[i + 1 - j])

        old_loc.append(0)
        if delta == 0:
            continue

        if len(old_loc) > len(err_loc):
            new_loc = gf_poly_scale(old_loc, delta)
            old_loc = gf_poly_scale(err_loc, gf_inverse(delta))
            err_loc = new_loc
        err_loc = gf_poly_add(err_loc, gf_poly_scale(old_loc, delta))

    while len(err_loc) > 1 and err_loc[0] == 0:
        err_loc.pop(0)

    errs = len(err_loc) - 1
    if errs * 2 > nsym:
        raise ReedSolomonDecodeError("too many Reed-Solomon symbol errors")
    return err_loc


def rs_find_errors(err_loc: Sequence[int], nmess: int) -> List[int]:
    errs = len(err_loc) - 1
    err_pos: List[int] = []

    for i in range(nmess):
        if gf_poly_eval(err_loc, gf_pow(GF_GENERATOR, i)) == 0:
            err_pos.append(nmess - 1 - i)

    if len(err_pos) != errs:
        raise ReedSolomonDecodeError("Reed-Solomon error locator search failed")
    return err_pos


def rs_correct_errata(codeword: Sequence[int], synd: Sequence[int], err_pos: Sequence[int]) -> List[int]:
    msg_out = list(codeword)
    n_errors = len(err_pos)
    if n_errors == 0:
        return msg_out

    matrix: List[List[int]] = []
    rhs = list(synd[1:n_errors + 1])
    for row_idx in range(n_errors):
        row: List[int] = []
        for pos in err_pos:
            exponent = (len(codeword) - 1 - pos) * row_idx
            row.append(gf_pow(GF_GENERATOR, exponent))
        matrix.append(row)

    magnitudes = gf_matrix_solve(matrix, rhs)
    for pos, magnitude in zip(err_pos, magnitudes):
        msg_out[pos] ^= magnitude

    return msg_out


def rs_decode_block(codeword: Sequence[int]) -> Tuple[List[int], int]:
    if len(codeword) != RS_N:
        raise ValueError(f"RS(255,127) decode expects {RS_N} bytes, got {len(codeword)}")

    synd = rs_calc_syndromes(codeword, RS_NSYM)
    if max(synd) == 0:
        return list(codeword[:RS_K]), 0

    err_loc = rs_find_error_locator(synd, RS_NSYM)
    err_pos = rs_find_errors(err_loc[::-1], len(codeword))
    corrected = rs_correct_errata(codeword, synd, err_pos)
    if max(rs_calc_syndromes(corrected, RS_NSYM)) > 0:
        raise ReedSolomonDecodeError("Reed-Solomon decoder failed syndrome check")

    corrected_symbols = sum(int(a != b) for a, b in zip(codeword, corrected))
    return corrected[:RS_K], corrected_symbols


def pack_bits_to_bytes(bits: Sequence[int]) -> Tuple[bytes, int]:
    dropped = len(bits) % 8
    usable = len(bits) - dropped
    out = bytearray()
    for i in range(0, usable, 8):
        value = 0
        for bit in bits[i:i + 8]:
            value = (value << 1) | int(bit)
        out.append(value)
    return bytes(out), dropped


def unpack_bytes_to_bits(data: bytes, n_bits: int | None = None) -> List[int]:
    bits = [int(bit) for byte in data for bit in f"{byte:08b}"]
    if n_bits is not None:
        bits = bits[:n_bits]
    return bits


def rs_encode_bytes(data: bytes) -> Tuple[bytes, RSChunkedReport]:
    if not data:
        return b"", RSChunkedReport(
            blocks_total=0,
            blocks_decoded=0,
            blocks_failed=0,
            blocks_truncated=0,
            corrected_symbol_errors=0,
            padding_bytes=0,
        )

    padding = (-len(data)) % RS_K
    padded = data + (b"\x00" * padding)
    blocks = [padded[i:i + RS_K] for i in range(0, len(padded), RS_K)]
    encoded = bytearray()
    for block in blocks:
        encoded.extend(rs_encode_block(block))

    return bytes(encoded), RSChunkedReport(
        blocks_total=len(blocks),
        blocks_decoded=len(blocks),
        blocks_failed=0,
        blocks_truncated=0,
        corrected_symbol_errors=0,
        padding_bytes=padding,
    )


def rs_decode_bytes(encoded: bytes, original_n_bytes: int | None = None) -> Tuple[bytes, RSChunkedReport]:
    if not encoded:
        return b"", RSChunkedReport(
            blocks_total=0,
            blocks_decoded=0,
            blocks_failed=0,
            blocks_truncated=0,
            corrected_symbol_errors=0,
            padding_bytes=0,
        )

    decoded = bytearray()
    blocks_total = 0
    blocks_decoded = 0
    blocks_failed = 0
    blocks_truncated = 0
    corrected_symbols = 0

    for offset in range(0, len(encoded), RS_N):
        block = encoded[offset:offset + RS_N]
        if not block:
            continue
        blocks_total += 1
        if len(block) < RS_N:
            blocks_truncated += 1
            decoded.extend(block[:min(RS_K, len(block))])
            continue

        try:
            msg, fixed = rs_decode_block(block)
            decoded.extend(msg)
            corrected_symbols += fixed
            blocks_decoded += 1
        except ReedSolomonDecodeError:
            blocks_failed += 1
            decoded.extend(block[:RS_K])

    padding = 0
    if original_n_bytes is not None:
        if len(decoded) > original_n_bytes:
            padding = len(decoded) - original_n_bytes
            decoded = decoded[:original_n_bytes]

    return bytes(decoded), RSChunkedReport(
        blocks_total=blocks_total,
        blocks_decoded=blocks_decoded,
        blocks_failed=blocks_failed,
        blocks_truncated=blocks_truncated,
        corrected_symbol_errors=corrected_symbols,
        padding_bytes=padding,
    )
