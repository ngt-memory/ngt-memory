"""
Безопасная персистентность NGT Memory (v2 формат).

Заменяет pickle (torch.save/load с weights_only=False) на:
  - safetensors — для всех тензоров (embeddings записей и концептов).
    Формат не исполняет код при загрузке, в отличие от pickle.
  - JSON — для всей структурированной метаинформации (тексты, граф,
    профиль, история чата, статистика).

Файлы сессии:
  {base}.memory.safetensors  — тензоры
  {base}.memory.json         — entries/concepts/graph метаданные
  {base}.session.json        — профиль, история чата, статистика

Версионирование: каждый JSON содержит "format_version". При несовместимой
мажорной версии загрузка отклоняется с понятной ошибкой (вместо тихой
порчи данных). Легаси .pt-файлы поддерживаются на чтение через
load_legacy_pt (с явным предупреждением в лог) — для миграции существующих
инсталляций; запись всегда идёт в новый формат.

Профиль сериализуется в данные (dict), а не как pickle-объект — изменение
класса UserProfile больше не ломает старые файлы.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch

logger = logging.getLogger("ngt_api.persistence")

FORMAT_VERSION = "2.0"
_COMPATIBLE_MAJOR = "2"

try:
    from safetensors.torch import save_file as _st_save, load_file as _st_load
    SAFETENSORS_AVAILABLE = True
except ImportError:  # pragma: no cover
    SAFETENSORS_AVAILABLE = False


class PersistenceFormatError(RuntimeError):
    """Несовместимая версия формата файлов сессии."""


def _check_version(meta: Dict, path: Path) -> None:
    v = str(meta.get("format_version", "?"))
    if v.split(".")[0] != _COMPATIBLE_MAJOR:
        raise PersistenceFormatError(
            f"{path}: format_version={v}, ожидается {_COMPATIBLE_MAJOR}.x. "
            f"Файл создан несовместимой версией NGT Memory."
        )


# ── UserProfile ↔ dict ───────────────────────────────────────────────

def profile_to_dict(profile) -> Dict:
    """Сериализует UserProfile в plain-данные (без pickle объектов)."""
    slots = {}
    for name, slot in profile.slots.items():
        if not slot.is_set:
            continue
        slots[name] = {
            "value": slot.value,
            "confidence": slot.confidence,
            "updated_at": slot.updated_at,
            "source": slot.source,
            "history": [
                {
                    "old_value": h.old_value,
                    "new_value": h.new_value,
                    "timestamp": h.timestamp,
                    "reason": h.reason,
                }
                for h in slot.history
            ],
        }
    return {"slots": slots, "explicit_facts": list(profile.explicit_facts)}


def profile_from_dict(data: Dict):
    """Восстанавливает UserProfile из plain-данных."""
    from ngt.core.user_profile import UserProfile, SlotChange

    profile = UserProfile()
    for name, sdata in (data.get("slots") or {}).items():
        slot = profile.slots.get(name)
        if slot is None:
            continue  # слот удалён в новой версии — пропускаем молча
        slot.value = sdata.get("value")
        slot.confidence = sdata.get("confidence", 0.0)
        slot.updated_at = sdata.get("updated_at", 0.0)
        slot.source = sdata.get("source", "")
        slot.history = [
            SlotChange(
                old_value=h.get("old_value"),
                new_value=h.get("new_value"),
                timestamp=h.get("timestamp", 0.0),
                reason=h.get("reason", ""),
            )
            for h in sdata.get("history", [])
        ]
    profile.explicit_facts = list(data.get("explicit_facts") or [])
    return profile


# ── NGTMemoryForLLM → файлы ──────────────────────────────────────────

def save_memory(memory, base_path: Union[str, Path]) -> None:
    """Сохраняет NGTMemoryForLLM в {base}.memory.safetensors + {base}.memory.json."""
    if not SAFETENSORS_AVAILABLE:
        raise RuntimeError(
            "safetensors не установлен — добавьте 'safetensors>=0.4' в зависимости. "
            "Запись в небезопасный pickle-формат отключена."
        )
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)

    tensors: Dict[str, torch.Tensor] = {}
    entries_meta = {}
    for eid, e in memory._entries.items():
        # .clone() обязателен: concept может шарить тот же storage, что и entry
        # (в store() cemb=embedding по умолчанию), а safetensors запрещает
        # сохранять тензоры с общей памятью.
        tensors[f"entry/{eid}"] = e.embedding.detach().cpu().clone().contiguous()
        entries_meta[str(eid)] = {
            "text": e.text,
            "metadata": e.metadata,
            "timestamp": e.timestamp,
            "importance": e.importance,
            "access_count": e.access_count,
            "concept_ids": e.concept_ids,
        }

    concepts_meta = {}
    for nid, c in memory.associations._id_to_concept.items():
        tensors[f"concept/{nid}"] = c.embedding.detach().cpu().clone().contiguous()
        concepts_meta[str(nid)] = {
            "name": c.name,
            "metadata": c.metadata,
            "created_at": c.created_at,
            "last_accessed": c.last_accessed,
            "access_count": c.access_count,
            "strength": c.strength,
        }

    meta = {
        "format_version": FORMAT_VERSION,
        "saved_at": time.time(),
        "embedding_dim": memory.embedding_dim,
        "max_entries": memory.max_entries,
        "entries": entries_meta,
        "next_entry_id": memory._next_entry_id,
        "concepts": concepts_meta,
        "next_concept_id": memory.associations._next_id,
        "graph_edges": {f"{a},{b}": w for (a, b), w in memory.associations._edges.items()},
        "session_id": memory._session_id,
        "stats": memory.stats,
    }

    # Атомарная запись: tmp → rename, чтобы крэш не оставил полуфайл
    st_path = Path(str(base) + ".memory.safetensors")
    js_path = Path(str(base) + ".memory.json")
    st_tmp = st_path.with_suffix(st_path.suffix + ".tmp")
    js_tmp = js_path.with_suffix(js_path.suffix + ".tmp")

    if tensors:
        _st_save(tensors, str(st_tmp))
    else:
        # safetensors не сохраняет пустой dict — пишем маркер
        _st_save({"_empty": torch.zeros(1)}, str(st_tmp))
    js_tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    st_tmp.replace(st_path)
    js_tmp.replace(js_path)


def load_memory(base_path: Union[str, Path], device: str = "cpu"):
    """Загружает NGTMemoryForLLM из v2-файлов. Возвращает память или None,
    если файлов нет. Бросает PersistenceFormatError при несовместимости."""
    from ngt.core.llm_memory import NGTMemoryForLLM
    from ngt.core.memory_entry import MemoryEntry
    from ngt.core.association_graph import ConceptNode

    base = Path(base_path)
    js_path = Path(str(base) + ".memory.json")
    st_path = Path(str(base) + ".memory.safetensors")
    if not js_path.exists():
        return None

    meta = json.loads(js_path.read_text(encoding="utf-8"))
    _check_version(meta, js_path)

    tensors = _st_load(str(st_path), device=device) if st_path.exists() else {}

    memory = NGTMemoryForLLM(
        embedding_dim=meta["embedding_dim"],
        max_entries=meta.get("max_entries", 10000),
        device=device,
    )

    for eid_str, edata in meta.get("entries", {}).items():
        eid = int(eid_str)
        emb = tensors.get(f"entry/{eid}")
        if emb is None:
            logger.warning("entry %s: embedding отсутствует в safetensors — пропуск", eid)
            continue
        entry = MemoryEntry(
            entry_id=eid,
            text=edata["text"],
            embedding=emb,
            metadata=edata.get("metadata", {}),
            importance=edata.get("importance", 1.0),
            concept_ids=edata.get("concept_ids", []),
        )
        entry.timestamp = edata.get("timestamp", time.time())
        entry.access_count = edata.get("access_count", 0)
        memory._entries[eid] = entry
    memory._next_entry_id = meta.get("next_entry_id", 0)
    memory._entry_id_list = list(memory._entries.keys())
    memory._emb_buffer = []
    memory._index_dirty = True

    for eid, entry in memory._entries.items():
        for cid in entry.concept_ids:
            memory._concept_to_entries.setdefault(cid, []).append(eid)

    for nid_str, cdata in meta.get("concepts", {}).items():
        nid = int(nid_str)
        emb = tensors.get(f"concept/{nid}")
        if emb is None:
            continue
        concept = ConceptNode(
            node_id=nid, name=cdata["name"], embedding=emb,
            metadata=cdata.get("metadata", {}),
        )
        concept.created_at = cdata.get("created_at", time.time())
        concept.last_accessed = cdata.get("last_accessed", time.time())
        concept.access_count = cdata.get("access_count", 1)
        concept.strength = cdata.get("strength", 1.0)
        memory.associations._id_to_concept[nid] = concept
        memory.associations._name_to_id[concept.name] = nid
        memory.associations._ensure_emb_capacity(nid + 1)
        memory.associations._embeddings[nid] = concept.embedding
        memory.associations._active_ids.append(nid)
    memory.associations._next_id = meta.get("next_concept_id", 0)
    memory.associations._emb_dirty = True

    for key_str, w in meta.get("graph_edges", {}).items():
        a_str, b_str = key_str.split(",")
        a, b = int(a_str), int(b_str)
        memory.associations._edges[(a, b)] = w
        memory.associations._adj.setdefault(a, {})[b] = w
        memory.associations._adj.setdefault(b, {})[a] = w

    memory._session_id = meta.get("session_id", 0)
    memory.stats = meta.get("stats", memory.stats)
    return memory


# ── Session-state (профиль/история/статистика) ───────────────────────

def save_session_state(base_path: Union[str, Path], profile, chat_history, stats) -> None:
    base = Path(base_path)
    payload = {
        "format_version": FORMAT_VERSION,
        "saved_at": time.time(),
        "profile": profile_to_dict(profile),
        "chat_history": chat_history,
        "stats": stats,
    }
    path = Path(str(base) + ".session.json")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_session_state(base_path: Union[str, Path]) -> Optional[Tuple]:
    """Возвращает (profile, chat_history, stats) или None если файла нет."""
    path = Path(str(base_path) + ".session.json")
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    _check_version(data, path)
    profile = profile_from_dict(data.get("profile") or {})
    return profile, data.get("chat_history", []), data.get("stats")


# ── Легаси (.pt) — только чтение, для миграции ───────────────────────

def legacy_pt_exists(base_path: Union[str, Path]) -> bool:
    return Path(str(base_path) + ".memory.pt").exists()


def load_legacy_pt(base_path: Union[str, Path], wrapper) -> bool:
    """Читает старый pickle-формат для миграции существующих сессий.

    ВНИМАНИЕ: torch.load(weights_only=False) исполняет произвольный код
    при десериализации. Использовать только для файлов, созданных этой
    же инсталляцией. После загрузки сессия будет пересохранена в v2.
    """
    from ngt.core.llm_memory import NGTMemoryForLLM
    from ngt.core.user_profile import UserProfile

    base = Path(base_path)
    mem_path = Path(str(base) + ".memory.pt")
    sess_path = Path(str(base) + ".session.pt")
    if not mem_path.exists():
        return False

    logger.warning(
        "loading legacy pickle session %s — будет мигрирована в safetensors+json "
        "при следующем сохранении", base.name,
    )
    wrapper.memory = NGTMemoryForLLM.load(mem_path)
    if sess_path.exists():
        state = torch.load(str(sess_path), map_location="cpu", weights_only=False)
        wrapper.profile = state.get("profile") or UserProfile()
        wrapper._chat_history = state.get("chat_history", [])
        wrapper._stats = state.get("stats", wrapper._stats)
    return True
