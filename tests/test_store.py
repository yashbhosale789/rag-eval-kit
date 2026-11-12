from rag_eval_kit import store


def test_content_id_ignores_crlf(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text('[{"id": "Q1"}]\n')
    b.write_bytes(b'[{"id": "Q1"}]\r\n')
    assert store.content_id(a) == store.content_id(b)
    assert len(store.content_id(a)) == 10


def test_content_id_is_key_order_independent(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text('[{"id": "Q1", "x": 1}]')
    b.write_text('[{"x": 1, "id": "Q1"}]')
    assert store.content_id(a) == store.content_id(b)


def test_load_skips_torn_line(tmp_path):
    p = tmp_path / "r.jsonl"
    store.append(p, {"key": "k1"})
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"key": "k2"')
    assert list(store.load(p)) == ["k1"]


def test_key_and_context_hash():
    assert store.key("baseline", "Q1", "v1") == "baseline|Q1|v1"
    assert store.context_hash(["a", "b"]) != store.context_hash(["b", "a"])
