"""Single hardened source for SurrealDB record-id sanitisation (W7.3 / BUG-10).

Every mixin in ``surreal_memory.storage.surrealdb`` that inlines a caller-supplied
id into a record literal or raw SurQL string MUST import ``_to_surreal_id`` (and,
for brain ids, ``_safe_brain_id``) from this module instead of defining its own
copy. Before BUG-10 this logic was duplicated across 13 files; 12 of those copies
were the pre-BUG-8 unhardened form (``record_id.replace("-", "_")``, with three
also stripping a table prefix first), so a hostile id reaching any of those
mixins could break out of the surrounding string/record literal. Consolidating
to one module makes the "single choke-point, no un-sanitised id reaches the
engine" guarantee literally true instead of merely aspirational.

This module must not import from ``store``, any mixin, or any other
``surreal_memory`` package — stdlib only — so it can be imported by every mixin
(including ``store.py`` itself) with zero risk of a circular import.
"""

from __future__ import annotations


def _to_surreal_id(record_id: str) -> str:
    """Convert a record ID to a valid SurrealDB record name (``[A-Za-z0-9_]``).

    Strips any existing table prefix (e.g. 'neuron:abc-123' -> 'abc_123')
    to prevent doubling when the caller later prepends 'neuron:', then maps
    every character outside ``[A-Za-z0-9_]`` to ``_``.

    SECURITY (deny-by-default injection guard): the result is inlined verbatim
    into record-id and query strings — ``neuron:{sid}`` (SurQL), the
    ``DELETE neuron_state:state_{sid}`` statement, and the ``eval::gql``
    SHORTEST fast-path in ``_get_path_gql`` (``{id:"{sid}"}``). Legitimate ids
    (UUID4, content-hashes) already live in this charset once '-' is folded to
    '_'; enforcing it here means a hostile ``source_id``/``target_id`` (e.g.
    from the REST ``GET /neurons/{source_id}/path`` route, whose value flows
    unresolved into ``get_path``) can never carry a ``"``, ``}``, ``)`` or
    whitespace that breaks out of the surrounding string/record literal to
    inject SurQL/GQL. Do NOT relax this back to a bare ``.replace('-', '_')``:
    the old form let ``x"}) ...`` reach the GQL parser once eval was enabled.
    """
    if ":" in record_id:
        record_id = record_id.rsplit(":", 1)[1]
    return "".join(
        c if ("a" <= c <= "z" or "A" <= c <= "Z" or "0" <= c <= "9" or c == "_") else "_"
        for c in record_id
    )


def _record_id_part(record_id: str) -> str:
    """Bare id part of a record id, as ``_to_surreal_id`` would have produced it.

    The inverse direction of the choke point above, and the reason it belongs
    here rather than being spelled out per mixin: SurrealDB renders a record id
    whose id part contains **no letter** in its quoted form —
    ``alerts:⟨1122334455667788⟩`` — and a plain ``split(":")[-1]`` hands those
    guillemets back to the caller. Feeding that back into ``_to_surreal_id``
    maps them to underscores, so the id no longer addresses its own row: the
    lookup misses and the caller is told the record does not exist.

    Measured on SurrealDB 3.2.0: the quoted form appears for ``0001``,
    ``1122334455667788``, ``12_34`` and ``_123``, but not for ``a1``,
    ``123abc``, ``1_2a`` or ``_a1`` — i.e. exactly when the id carries no
    letter. Ids minted as ``str(uuid4())`` almost always carry one; ids minted
    as ``uuid4().hex[:16]`` (alerts) are letter-free about once in 1150.
    """
    if ":" in record_id:
        record_id = record_id.rsplit(":", 1)[1]
    return record_id.strip("⟨⟩`")


def _safe_brain_id(brain_id: str) -> str:
    """Validate a brain id before it is inlined raw into a record id / SurQL.

    Brain ids legitimately contain '.' and '-' (e.g. 'my-brain.v2'), so unlike
    neuron ids they are NOT folded through _to_surreal_id. This is the
    store-layer choke point that makes the "no un-sanitised id reaches the
    engine" guarantee literally true: it fail-closed REJECTS any id carrying a
    quote / brace / paren / semicolon / whitespace / backtick / control char
    that could break out of the ``brain:{id}`` / ``device:{brain_id}_{did}``
    record literal or the raw ``UPDATE brain:{id} SET ...`` statement. Mirrors
    the REST ``_BRAIN_ID_PATTERN`` (``[A-Za-z0-9_.-]``, <=128) so the guarantee
    no longer depends on the REST layer having validated first.
    """
    if (
        not isinstance(brain_id, str)
        or not brain_id
        or len(brain_id) > 128
        or any(
            not ("a" <= c <= "z" or "A" <= c <= "Z" or "0" <= c <= "9" or c in "_.-")
            for c in brain_id
        )
    ):
        raise ValueError(f"unsafe brain id: {brain_id!r}")
    return brain_id
