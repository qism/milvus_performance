# milvus_performance
单节点性能测试： milvus 和 zilliz cloud
zilliz cloud有$100 credits 可试用

# pymilvus
version: 2.2.11

# 测试项：

## 纯search
## 一条search 一条insert：模拟流式数据实时插入场景

# 优化项：
search和insert可异步进行；集群多节点，不同节点分别处理search和insert

