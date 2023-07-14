import time
import argparse
from multiprocessing import Pool, current_process
import numpy as np
from loguru import logger
from pymilvus import utility
from pymilvus import connections
from pymilvus import Collection

parser = argparse.ArgumentParser()
parser.add_argument('--process_num', type=int, default=1)
parser.add_argument('--vector_num', type=int, default=10000)
parser.add_argument("--collection_name", type=str, default="collection_s2_m16_f128")
parser.add_argument("--collection_alias", type=str, default="performance_test")
args = parser.parse_args()

MILVUS_DEFAULT_ALIAS = args.collection_alias
MILVUS_COLLECTION_NAME = args.collection_name



# prepare query
logger.info('begin to load vectors')
array = np.load('data.npz')
vector_list = array['vector']

queries = []
for i in range(args.vector_num):
    dic = {
        'embedding_vector': vector_list[i].tolist(),
    }
    queries.append(dic)
logger.info(f'finish loading vectors.Total num is {len(queries)}')


# zilliz cloud
user = ''
password = ''
endpoint = ''

# milvus
# MILVUS_DEFAULT_PORT = "19530"
# HOST = 'localhost'

def init_pool():
    alias = current_process().name
    global milvus_conn
    milvus_conn = connections.connect(
        alias=alias,
        uri=endpoint, #  Public endpoint obtained from Zilliz Cloud
        secure=True,
        user=user, # Username specified when you created this database 
        password=password # Password specified when you created this database
    )
    # milvus
#   milvus_conn = connections.connect(
#         alias=MILVUS_DEFAULT_ALIAS,
#         host=HOST,
#         port=MILVUS_DEFAULT_PORT
#     )
    global milvus_collection
    milvus_collection = Collection(
        name=MILVUS_COLLECTION_NAME, using=alias
    )

#  多进程 查询测试
def milvus_search(query_vector, idx):
    global milvus_collection
    param = {"metric_type": "IP", "params": {"ef": 128}}
    search_param = {
            "data": [query_vector],
            "anns_field": "vector",
            "param": param,
            "limit": 20
        }
    start_time = time.time()
    _ = milvus_collection.search(**search_param)
    end_time = time.time()
    return (end_time - start_time) * 1000

if __name__ == '__main__':
    results = []

    logger.info(f'begin to search with {args.process_num} processes.')
    vector_num = args.vector_num
    start_time = time.time()

    pool = Pool(processes=args.process_num, initializer=init_pool)
    for idx, item in enumerate(queries):
        query = item['embedding_vector']
        result = pool.apply_async(milvus_search, (query, idx))
        results.append(result)

    pool.close()
    pool.join()
    
    # time analysis
    all_time = []
    for one in results:
        one_time = (one.get())
        all_time.append(one_time)

    import pandas as pd
    df_search_time = pd.DataFrame({'search_time': all_time})

    logger.info('finish.')
    logger.info(f'完成检索平均时间:{(time.time()-start_time)*1000/vector_num:.2f}ms')
    logger.info(f'search avg, p99, p999:{df_search_time.mean()[0]:.2f}ms, {df_search_time.quantile(0.99)[0]:.2f}ms, {df_search_time.quantile(0.999)[0]:.2f}ms')
