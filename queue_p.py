import heapq 
import itertools

class PriorityQueue:
    def __init__(self, elements = None):
        if(elements == None):
            self.elements = []
        elif type(elements) == list:
            heapq.heapify(elements)
            self.elements = elements

    def __str__(self):
        return str(self.elements)

    # for checking if the queue is empty
    def isEmpty(self):
        return len(self.elements) == 0

    # for inserting an element in the queue
    def push(self, element):
        heapq.heappush(self.elements, element)
    # for popping an element based on Priority
    def pop(self):
        return heapq.heappop(self.elements)

#FIFO Queue Implementation
class queue:
    def __init__(self):
        self.items = []

    def __iter__(self):
        for i in self.items:
            yield i

    def Empty(self):
        return self.items == []

    def push(self,item):
        self.items.insert(0,item)

    def pop(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

    def front(self):
        return self.items[len(self.items)-1]
    def in_q(self,item):
        if item in self.items:
            return 1
        else:
            return 0
    def index(self,item):
#         print(self.items.index(item))
        return self.items.index(item)