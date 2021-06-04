import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from transformers import PreTrainedTokenizer

class OWTC:
    def __init__(self, path_to_corpus: Path=None, shard: int=None):
        """
        OpenWebText Module
        
        WARNING: `load_corpus` loads the entire OpenWebText *into memory*, so make sure you have at least 200GB of RAM :)
        If you don't have this amount of memory, make sure to sample a subset of the corpus (see below)
        """
        self.path_to_corpus = path_to_corpus
        self.shard_paths = list(self.path_to_corpus.iterdir())
        if shard is not None:
            self.shard_paths = [self.shard_paths[shard]]
        self.num_shards = len(self.shard_paths)
        self.corpus = None

    @staticmethod
    def query(query: str, db_path: Path, batch_size: int=None):
        """
        Query database
        
        Params
        ------
        query: str => SQL query string
        db_path: Path => path to database
        batch_size: int => batch size to perform query. if not set, will run entire sql query at once with no chunking.
        """
        with sqlite3.connect(db_path) as conn:
            if batch_size:
                dfs = pd.read_sql_query(query, conn, chunksize=batch_size)
            else:
                dfs = [pd.read_sql_query(query, conn)]
        out = []
        for df in dfs:
            out.append(df)
        return pd.concat(out, 0)
        
    def _sample_shard(self, shard_path: Path, sample: int, batch_size: int=None) -> pd.DataFrame:
        """
        Sample a database shard
        
        Params
        ------
        sample: int => sample size
        shard_path: Path => path to database shard
        batch_size: int => batch size to perform query. if not set, will run entire sql query at once with no chunking.
        """
        query = f"SELECT id, md5_hash, text, url, subreddit, karma from docs order by RANDOM() limit {sample}"
        return self.query(query, shard_path, batch_size)
    
    def _query_shard(self, shard_path: Path, batch_size: int=None, low_id: int=None, high_id: int=None) -> pd.DataFrame:
        """
        Query a database shard for data between a range of ids. 
                
        Params
        ------
        shard_path: Path => path to database shard
        batch_size: int => batch size to perform query. if not set, will run entire sql query at once with no chunking.
        low_id: int => lower bound of ids requested
        high_id: int => upper bound of ids requested
        """
        query = f"SELECT id, md5_hash, text, url, subreddit, karma from docs where id between {low_id} and {high_id}"
        return self.query(query, shard_path, batch_size)

    def _get_min_id(self, shard_path: Path) -> int:
        """
        Get the minimum ID for a database shard
        
        Params
        ------
        shard_path: Path => path to database shard
        """
        with sqlite3.connect(shard_path) as conn:
            query = f"SELECT min(id) from docs"
            min_id = pd.read_sql_query(query, conn)['min(id)'][0]
        return min_id

    def load_corpus(self, batch_size: int=None, sample: int=None) -> None:
        """
        Load openwebtext corpus into memory
        
        Params
        ------
        batch_size: int => batch size to use when loading the corpus. If not set, will load entire corpus at once.
        sample: int => if set, will subsample corpus to provided size (random sample)
        """
        if sample: 
            dfs = Parallel(n_jobs=len(self.shard_paths))(delayed(self._sample_shard)(shard_path, 
                                                                                     sample // self.num_shards,
                                                                                     batch_size) 
                                                for ix, shard_path in tqdm(enumerate(self.shard_paths), total=self.num_shards, desc='loading corpus'))
            self.corpus =  pd.concat(dfs,0).drop_duplicates(subset=['id']).sort_values(by='id')
        else:
            min_ids = Parallel(n_jobs=len(self.shard_paths))(delayed(self._get_min_id)(i) for i in self.shard_paths)
            ranges = np.linspace(0, 1400000, 10)
            out = []
            for low, high in tqdm(zip(ranges[:-1], ranges[1:]), total=len(ranges)-1, desc='loading corpus'):
                dfs = Parallel(n_jobs=len(self.shard_paths))(delayed(self._query_shard)(shard_path,
                                                                            batch_size, 
                                                                            min_ids[ix] + int(low), 
                                                                            min_ids[ix] + int(high)) 
                                                        for ix, shard_path in enumerate(self.shard_paths))
                out.append(dfs)
            self.corpus = pd.concat([pd.concat(x, 0) for x in out],0).drop_duplicates(subset=['id']).sort_values(by='id')

        return
