from lib.schemas import EmbeddingRecord, Neighbor
from lib.services.similarity_service import SimilarityService


class FakeStore:
    def __init__(self, records):
        self.records = records

    def all(self):
        return self.records

    def append(self, record):
        self.records.append(record)


def make_service(records=None, metric="cosine", threshold=0.5):
    return SimilarityService(
        store=FakeStore(records or []),
        similarity_metric=metric,
        similarity_threshold=threshold,
        top_k=10,
        image_size=224,
        model_name="baseline",
    )


def rec(breed, emb):
    return EmbeddingRecord(id_imagen=breed + str(emb), embedding=emb, path=f"{breed}.jpg", breed=breed)


def test_search_sorted_desc_and_limited():
    svc = make_service([rec("a", [1.0, 0.0]), rec("b", [0.0, 1.0]), rec("c", [0.9, 0.1])])
    res = svc.search_similar_images([1.0, 0.0], 2)
    assert [n.breed for n in res] == ["a", "c"]
    assert res[0].score >= res[1].score


def test_search_l2_metric():
    svc = make_service([rec("a", [1.0, 0.0]), rec("b", [0.0, 1.0])], metric="l2")
    assert svc.search_similar_images([1.0, 0.0], 5)[0].breed == "a"


def test_predict_empty_is_unknown():
    assert make_service().predict_breed_from_neighbors([]) == ("unknown", 0.0)


def test_predict_weighted_vote():
    svc = make_service()
    nb = [Neighbor(path="1", breed="x", score=0.9), Neighbor(path="2", breed="y", score=0.6),
          Neighbor(path="3", breed="y", score=0.6)]
    breed, score = svc.predict_breed_from_neighbors(nb)
    assert breed == "y" and abs(score - 0.6) < 1e-9  # 1.2 > 0.9


def test_predict_below_threshold_unknown():
    svc = make_service(threshold=0.8)
    breed, _ = svc.predict_breed_from_neighbors([Neighbor(path="1", breed="x", score=0.5)])
    assert breed == "unknown"
