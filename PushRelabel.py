# push-relabel algorithm


def MaxFlow(C, s, t):
    n = len(C)  # C is the capacity matrix
    F = [[0] * n for i in range(n)]  # initialize preflow

    # the residual capacity from u to v is C[u][v] - F[u][v]
    height = [0] * n  # height of node
    excess = [0] * n  # flow into node minus flow from node
    seen = [0] * n  # neighbours seen since last relabel

    # node "queue"
    node_list = [i for i in range(n) if i != s and i != t]

    # push operation
    def push(u, v):
        send = min(excess[u], C[u][v] - F[u][v])
        F[u][v] += send
        F[v][u] -= send
        excess[u] -= send
        excess[v] += send

    # relabel operation
    def relabel(u):
        # find smallest new height making a push possible,
        # if such a push is possible at all
        min_height = float('inf')

        for v in range(n):
            if C[u][v] - F[u][v] > 0:
                min_height = min(min_height, height[v])
                height[u] = min_height + 1

    def discharge(u):
        while excess[u] > 0:
            if seen[u] < n:  # check next neighbour
                v = seen[u]
                if C[u][v] - F[u][v] > 0 and height[u] > height[v]:  # admissible edge
                    push(u, v)
                else:
                    seen[u] += 1
            else:  # we have checked all neighbours. must relabel
                relabel(u)
                seen[u] = 0

    # --- Initialization
    height[s] = n   # longest path from source to sink is less than n long

    # initialize nodes' excess
    # send as much flow as possible to neighbours of source
    excess[s] = float("inf")

    # initialize pre-flow of the edges start from node s
    # initialize excess of nodes of s's neighbors
    for v in range(n):
        push(s, v)

    p = 0
    while p < len(node_list):
        u = node_list[p]
        old_height = height[u]

        # relabel
        discharge(u)

        if height[u] > old_height:
            node_list.insert(0, node_list.pop(p))  # move to front of list
            p = 0  # re-start from front of the list
        else:  # move to next node
            p += 1  

    return sum(F[s])


#make a capacity graph
#node s  o  p  q  r  t
C = [[0, 3, 3, 0, 0, 0],  # s
     [0, 0, 2, 3, 0, 0],  # o
     [0, 0, 0, 0, 2, 0],  # p
     [0, 0, 0, 0, 4, 2],  # q
     [0, 0, 0, 0, 0, 2],  # r
     [0, 0, 0, 0, 0, 3]]  # t

source = 0  # A
sink = 5    # F
max_flow_value = MaxFlow(C, source, sink)
print("Push-Relabeled(Preflow-push) algorithm")
print("max_flow_value is: ", max_flow_value)
