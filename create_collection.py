import pickle
import argparse
import numpy as np
from loguru import logger
from pymilvus import connections
from pymilvus import CollectionSchema, FieldSchema, DataType
from pymilvus import Collection
from pymilvus import utility


parser = argparse.ArgumentParser()
parser.add_argument("--shards", type=int, default=2)
parser.add_argument("--m", type=int, default=16)
parser.add_argument("--f", type=int, default=128)
parser.add_argument("--collection_name", type=str, default="collection_s2_m16_f128")
parser.add_argument("--collection_alias", type=str, default="performance_test")
args = parser.parse_args()

MILVUS_DEFAULT_ALIAS = args.collection_alias
MILVUS_COLLECTION_NAME = args.collection_name
logger.info(f"Collection name: {MILVUS_COLLECTION_NAME}")


# zilliz cloud
user = ''
password = ''
endpoint = ''

# milvus
# MILVUS_DEFAULT_PORT = "19530"
# HOST = 'localhost'

# prepare schema
idx = FieldSchema(
	        name="id",
	        dtype=DataType.INT64,
	        is_primary=True,
	        )

vector = FieldSchema(
	        name="vector",
	        dtype=DataType.FLOAT_VECTOR,
	        dim=52,
	        )

fields = [idx, vector]
schema = CollectionSchema(
	fields=fields,
	description=MILVUS_COLLECTION_NAME,
	enable_dynamic_field=True
)

# vector index paras
index_params = {
            "metric_type": "IP",
            "index_type": "HNSW",
            "params": {"efConstruction":args.f, "M": args.m},
        }

# connection
# zilliz cloud
connections.connect(
        alias=MILVUS_DEFAULT_ALIAS,
        uri=endpoint, #  Public endpoint obtained from Zilliz Cloud
        # token=token,
        secure=True,
        user=user, # Username specified when you created this database 
        password=password # Password specified when you created this database
    )
# milvus
# connections.connect(
#         alias=MILVUS_DEFAULT_ALIAS,
#         host=HOST,
#         port=MILVUS_DEFAULT_PORT
#     )
logger.info("connect successfully!")
logger.info("Creating collection: ")

# create collection
if utility.has_collection(MILVUS_COLLECTION_NAME, using=MILVUS_DEFAULT_ALIAS):
	utility.drop_collection(MILVUS_COLLECTION_NAME, using=MILVUS_DEFAULT_ALIAS)

collection = Collection(
            name=MILVUS_COLLECTION_NAME,
            schema=schema,
            using=MILVUS_DEFAULT_ALIAS,
            shards_num=args.shards
)

logger.info("Creating collection successfully! ")

# upload data
logger.info("upload data: ")
import numpy as np
array = np.load('data.npz')

# bulk
batch = 100000
num = 3000000/ batch

for idx in range(int(num)):
    data = [
        [i for i in range(idx*batch, (idx+1)*batch,1)],
        [i for i in array['vector'][idx*batch:(idx+1)*batch]]
    ]
    mr = collection.insert(data)
    logger.info(f"{idx} batch is done!")

logger.info("upload data successfully! ")
logger.info("Creating collection: ")

# create index
collection.create_index(field_name="vector", index_params=index_params)

# other fileds 
# for field_schema in collection.schema.fields:
#     if field_schema.name in ["id", "vector"]:
#         continue
#     collection.create_index(field_name=field_schema.name, index_name=field_schema.name)

logger.info("Creating collection successfully! ")

collection.load()
collection.flush()
print("The num_entities of collection is ", collection.num_entities)
