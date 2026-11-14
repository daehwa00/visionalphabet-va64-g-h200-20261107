from types import SimpleNamespace
import requests
import pytest
import download

def test_retries_truncated_chunk_and_returns_only_complete(monkeypatch):
    replies=[b'bad',b'complete']
    class Session:
        def get(self,*a,**kw):
            assert kw['headers']['Accept']=='application/octet-stream'
            return SimpleNamespace(status_code=200,content=replies.pop(0),raise_for_status=lambda:None)
        def close(self):pass
    monkeypatch.setattr(download.requests,'Session',Session)
    monkeypatch.setattr(download.time,'sleep',lambda _:None)
    r=download.BinaryReader(SimpleNamespace(base='https://invalid',token='test',session='test'))
    assert r.get('/chunk',8)==b'complete'
    assert not replies

def test_incomplete_chunk_never_returned(monkeypatch):
    class Session:
        def get(self,*a,**kw):return SimpleNamespace(status_code=200,content=b'bad',raise_for_status=lambda:None)
        def close(self):pass
    monkeypatch.setattr(download.requests,'Session',Session)
    monkeypatch.setattr(download.time,'sleep',lambda _:None)
    r=download.BinaryReader(SimpleNamespace(base='https://invalid',token='test',session='test'))
    with pytest.raises(RuntimeError,match='retries exhausted'):r.get('/chunk',8)
