import heapq#采用堆排序的一种工具
map_item_score={2:0.1,1:0.2,4:0.3,5:0.5,7:0.6}
a=heapq.nlargest(3, map_item_score, key=map_item_score.get)
#top3为[0.6,0.5,0.3]，对应的key为[7,5,4]
print(a)
