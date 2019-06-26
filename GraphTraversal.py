# coding=utf-8

# undirected graph(无向图)邻接表
G_ug = {
    's': set(['w', 'r']),
    'r': set(['s', 'v']),
    'w': set(['s', 't', 'x']),
    't': set(['w', 'x', 'u']),
    'x': set(['w', 't', 'u', 'y']),
    'u': set(['t', 'x', 'y']),
    'y': set(['x', 'u']),
    'v': set(['r'])
}  # 

# 邻接表
G_dg = {
    'u': set(['v', 'x']),
    'v': set(['y']),
    'x': set(['v']),
    'y': set(['x']),
    'w': set(['y', 'z']),
    'z': set(['z'])  # self loop
}  # directed 

def print_path(G, node_set, s, v):
    """
    打印从s到v的路径
    """
    if s == v:
        print(s)
    elif node_set[v]['parent'] == None:
        print('=> no path from ', s, 'to', v, 'exists')
    else:
        print_path(G, node_set, s, node_set[v]['parent'])

from queue import Queue

def BFS(G, start):
    """
    图的广度优先遍历: 图的广度优先遍历是构建一颗广度优先树的过程
    广度优先遍历主要依靠队列Queue实现
    由于邻近节点访问顺序可能不同，因此广度优先访问顺序可能不一致
    """
    node_set = {}
    
    # 初始化每一个节点, 放入一个集合
    for key in G.keys():
        vertex = {}

        vertex['name'] = key
        vertex['color'] = 'WHITE'
        vertex['dist'] = 0
        vertex['parent'] = None

        node_set[key] = vertex
    
    queue = Queue()

    # 初始化开始节点的访问
    start_node = node_set[start]
    start_node['color'] = 'GRAY'
    start_node['dist'] = 0
    start_node['parent'] = None

    # 放入初始结点
    queue.put(start_node)

    while not queue.empty():
        vertex = queue.get()
        print('=>%s' % (vertex['name']), end='')

        for next in G[vertex['name']]:
            node = node_set[next]

            # 如果节点没有访问过
            if node['color'] == 'WHITE':
                node['color'] = 'GRAY'
                node['dist'] = vertex['dist'] + 1
                node['parent'] = vertex['name']

                queue.put(node)
        
        # 当前出队列结点vertex所有的邻接节点都已被访问
        # 标记为Black
        vertex['color'] = 'BLACK'

    print('\n\n', node_set, '\n')

    # print_path(G, node_set, 's', 'y')


def DFS(G, start):
    """
    图的深度优先遍历
    """
    def DFS_visit(G, node):
        """
        深度优先访问
        """
        global time, node_set

        # first detected
        time += 1
        node['detect'] = time
        node['color'] = 'GRAY'
        
        for neigh in G[node['name']]:
            node_neigh = node_set[neigh]

            if node_neigh['color'] == 'WHITE':
                node_neigh['parent'] = node['name']
                DFS_visit(G, node_neigh)
        
        # node finished
        node['color'] = 'BLACK'
        time += 1
        node['finish'] = time
        print('=>', node['name'])

    # --------- 初始化
    global node_set, time
    node_set = {}
    
    # 初始化每一个节点, 放入一个集合
    for key in G.keys():
        vertex = {}

        vertex['name'] = key
        vertex['color'] = 'WHITE'
        vertex['detect'] = 0
        vertex['finish'] = 0
        vertex['parent'] = None

        # 节点放入字典
        node_set[key] = vertex

    # 开始遍历
    time = 0
    for node_name in node_set.keys():
        node = node_set[node_name]
        if node['color'] == 'WHITE':
            DFS_visit(G, node)

    print('\n\n', node_set, '\n')


if __name__ == '__main__':
    BFS(G=G_ug, start='s')

    DFS(G=G_dg, start='u')
