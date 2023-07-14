import json
import  time
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


logger.info('begin to load vectors')
array = np.load('data.npz') # aws
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


# # 多进程 1w条测试
def milvus_insert_search(query_vector, idx):
    global milvus_collection
    s = time.time()
    param = {"metric_type": "IP", "params": {"ef": 128}}
    search_param = {
            "consistency_level": "Bounded",
            "data": [query_vector],
            "anns_field": "vector",
            "param": param,
            "limit": 20
        }
    data = [ [idx],
    [query_vector] ]
    # search
    s_start_time = time.time()
    _ = milvus_collection.search(**search_param)
    s_end_time = time.time()
    # insert
    _ = milvus_collection.insert(data)
    i_end_time = time.time()
    return {'id': idx, 'search_time': (s_end_time - s_start_time) * 1000, 'insert_time': (i_end_time - s_end_time) * 1000, 'task_time': (i_end_time - s) * 1000}

def write_file(filename, temp):
    with open(f'{filename}', 'w', encoding='utf-8') as fw:
        for one in temp:
            fw.write(json.dumps(one.get(), ensure_ascii=False)+'\n')
    logger.info(f'write file done. filename is {filename}')


if __name__ == '__main__':
    logger.info(f'begin to search with {args.process_num} processes.')
    vector_num = args.vector_num
    start_time = time.time()

    results = []
    pool = Pool(processes=args.process_num, initializer=init_pool)
    for idx, query in enumerate(queries):
        query_vector = query['embedding_vector']
        result = pool.apply_async(milvus_insert_search, (query_vector, idx))
        results.append(result)
        # record res
        if len(results) % 50000 == 0:
            filename = f'{idx}_res.json'
            write_file(filename, results)
            results = []

    pool.close()
    pool.join()
    logger.info('finish.')
    logger.info(f'{(time.time()-start_time)*1000/vector_num:.2f}ms')
    logger.info(f'{(time.time()-start_time):.2f}s')

